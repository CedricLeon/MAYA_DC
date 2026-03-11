"""MonitorValReconstruction callback for RCMC compression training.

Logs a 4-row x N-column figure to WandB at the end of the first validation
batch every ``log_every_n_epochs`` epochs:

    Row 0 - RCMC input         (buffer trimmed, log-intensity)
    Row 1 - RCMC reconstructed (x_hat buffer trimmed, log-intensity)
    Row 2 - SLC reconstructed  (after sarpyx azimuth focusing, log-intensity)
    Row 3 - SLC target         (ground truth, log-intensity)
"""

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from maya4 import GT_MAX, GT_MIN, RC_MAX, RC_MIN
from torch import Tensor

from src.utils.logging import print_images_statistics
from src.utils.sarpyx_azimuth_compression import full_azimuth_compress_batch

EPS = 1e-2
_ROW_TITLES = [
    "RCMC input\n(logI)",
    "RCMC reconstructed\n(logI)",
    "SLC reconstructed\n(logI)",
    "SLC target\n(logI)",
]


def _to_logI(t: Tensor) -> Tensor:
    """Convert ``(B, 2, H, W)`` real/imag tensor to ``(B, H, W)`` log-Intensity.

    ``log_intensity = log(real² + imag² + ε)``

    Returns a Tensor on the same device as the input.
    Only convert to numpy at matplotlib call sites.
    """
    return torch.log(t[:, 0] ** 2 + t[:, 1] ** 2 + EPS)  # (B, H, W)


def _minmax_denorm(t: Tensor, vmin: float, vmax: float) -> Tensor:
    """Invert MAYA4 minmax normalization: ``[0, 1] → [vmin, vmax]``."""
    return t * (vmax - vmin) + vmin


def _clip_mean_std(img: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Clip ``img`` to ``mean ± factor * std`` for better contrast in visualizations."""
    mu, sigma = img.mean(), img.std()
    return np.clip(img, mu - factor * sigma, mu + factor * sigma)


class MonitorValReconstruction(Callback):
    """Log RCMC / SLC reconstruction grids to WandB during validation.

    Produces a **4-row x ``num_images``-column** figure on the first validation
    batch every ``log_every_n_epochs`` epochs:

    * **Row 0** - RCMC input (core only, buffer trimmed)
    * **Row 1** - RCMC reconstructed (``x_hat``, buffer trimmed)
    * **Row 2** - SLC reconstructed (after sarpyx azimuth focusing)
    * **Row 3** - SLC target (ground truth)

    All rows are shown in **log-intensity** (``log(Re² + Im² + ε)``),
    contrast-clipped to ``mean ± clip_factor * std``.

    Args:
        log_every_n_epochs: How often (in epochs) to produce the figure.
        num_images: Number of patches (columns). Clamped to the actual batch size.
        clip_factor: Std-deviation multiplier for contrast clipping.
        verbose: Print per-image statistics to stdout.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        num_images: int = 3,
        clip_factor: float = 3.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images
        self.clip_factor = clip_factor
        self.verbose = verbose

    # ------------------------------------------------------------------
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Verify the callback is attached to the correct module class."""
        cls = pl_module.__class__.__name__
        if cls != "RCMCDCmodule":
            raise ValueError(
                f"MonitorValReconstruction expects RCMCDCmodule, got '{cls}'. "
                "Remove this callback or attach it to the correct module."
            )

    # ------------------------------------------------------------------
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Build and log the 4-row reconstruction figure for the first val batch."""
        if (trainer.current_epoch % self.log_every_n_epochs != 0) or batch_idx > 0:
            return

        # ── Unpack batch ──────────────────────────────────────────────
        if len(batch) == 5:
            rcmc_batch, slc_batch, metadata_list, ephemeris_list, _ = batch
        elif len(batch) == 2:
            rcmc_batch, slc_batch = batch
            metadata_list, ephemeris_list = None, None
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}, expected 2 or 5")

        buffer: int = pl_module.hparams.azimuth_buffer  # type: ignore[attr-defined]
        B, _C, Az, _Rg = rcmc_batch.shape
        Az_core = Az - 2 * buffer
        n = min(self.num_images, B)

        # ── Forward pass (no grad, on same device as model) ───────────
        with torch.no_grad():
            output = pl_module(rcmc_batch)

            # Denormalize x_hat to physical IQ scale before focusing so that
            # slc_recon is in the same physical intensity domain as slc_target.
            x_hat_phys = _minmax_denorm(output.x_hat, RC_MIN, RC_MAX)  # (B,2,Az+2*buf,Rg)

            # Azimuth-focus the denormalized reconstructed RCMC (buffer still attached)
            slc_recon = full_azimuth_compress_batch(
                x_hat_phys,
                metadata_list,
                ephemeris_list,
                buffer_size=buffer,
                device=str(output.x_hat.device),
            )  # (B, 2, Az_core, Rg)  — physical SLC scale

        # ── Trim azimuth buffer; denormalize to physical scale ────────
        rcmc_input_core = _minmax_denorm(
            rcmc_batch[:, :, buffer : buffer + Az_core, :], RC_MIN, RC_MAX
        )  # (B,2,Az_core,Rg) — physical RCMC
        rcmc_recon_core = _minmax_denorm(
            output.x_hat[:, :, buffer : buffer + Az_core, :], RC_MIN, RC_MAX
        )  # (B,2,Az_core,Rg) — physical RCMC
        slc_target_core = _minmax_denorm(
            slc_batch[:, :, buffer : buffer + Az_core, :], GT_MIN, GT_MAX
        )  # (B,2,Az_core,Rg) — physical SLC

        # ── Log-intensity Tensors on device (B, H, W) ─────────────────
        rows_data: list[Tensor] = [
            _to_logI(rcmc_input_core),
            _to_logI(rcmc_recon_core),
            _to_logI(slc_recon),
            _to_logI(slc_target_core),
        ]

        if self.verbose:
            print_images_statistics(
                {
                    title.split("\n")[0]: rows_data[i][0].cpu().numpy()
                    for i, title in enumerate(_ROW_TITLES)
                },
                title=f"[MonitorValReconstruction] Epoch {trainer.current_epoch} — patch 0 logI stats",
            )

        # ── Build figure ──────────────────────────────────────────────
        fig, axes = plt.subplots(4, n, figsize=(4 * n, 16), squeeze=False)

        for row_idx, (row_title, row_tensor) in enumerate(zip(_ROW_TITLES, rows_data)):
            for col_idx in range(n):
                img_np = _clip_mean_std(row_tensor[col_idx].cpu().numpy(), self.clip_factor)
                ax = axes[row_idx, col_idx]
                ax.imshow(img_np, cmap="gray", aspect="auto")
                ax.axis("off")
                if col_idx == 0:
                    # Row label on the left
                    ax.text(
                        -0.05,
                        0.5,
                        row_title,
                        transform=ax.transAxes,
                        fontsize=9,
                        rotation=90,
                        va="center",
                        ha="right",
                        weight="bold",
                    )
                if row_idx == 0:
                    ax.set_title(f"Patch {col_idx}", fontsize=9)

        # ── Summary metrics for the figure title ──────────────────────
        # MSE in logI between SLC recon and SLC target (rows 2 & 3)
        mse_slc = float(torch.mean((rows_data[2][:n] - rows_data[3][:n]) ** 2).cpu())
        # MAE in logI between RCMC recon and RCMC input (rows 1 & 0)
        mae_rcmc = float(torch.mean(torch.abs(rows_data[1][:n] - rows_data[0][:n])).cpu())

        fig.suptitle(
            f"Validation epoch {trainer.current_epoch} — "
            f"SLC logI MSE: {mse_slc:.4f}   RCMC logI MAE: {mae_rcmc:.4f}",
            fontsize=11,
        )
        plt.tight_layout()

        # ── Log to WandB ──────────────────────────────────────────────
        if (
            pl_module.logger is not None
            and hasattr(pl_module.logger, "experiment")
            and hasattr(pl_module.logger.experiment, "log")
        ):
            import wandb  # local import — optional dep

            pl_module.logger.experiment.log(  # type: ignore[attr-defined]
                {
                    "val_reconstructions": wandb.Image(fig),
                    "val_batch/slc_log_mse": mse_slc,
                    "val_batch/rcmc_log_mae": mae_rcmc,
                },
                step=trainer.global_step,
            )

        plt.close(fig)
