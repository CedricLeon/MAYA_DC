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
from maya4 import GT_MAX, GT_MIN, RC_MAX, RC_MIN, minmax_inverse, minmax_normalize
from torch import Tensor

from src.models.components.losses import (
    complex_coherence_loss,
    complex_correlation_metric,
    kde_histogram_loss,
    phase_preservation_metric,
    psnr_amplitude,
    ssim_amplitude,
)
from src.utils.logging import print_images_statistics
from src.utils.processing_utils import EPS, clip_mean_std_numpy, phys_to_logI_torch
from src.utils.sarpyx_azimuth_compression import full_azimuth_compress_batch


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

        self.ROW_TITLES = [
            "RCMC input\n(logI)",
            "RCMC reconstructed\n(logI)",
            "SLC reconstructed\n(logI)",
            "SLC target\n(logI)",
        ]

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
        rcmc_batch, slc_batch, metadata_list, ephemeris_list, coords_list = batch

        az_buffer: int = pl_module.hparams.azimuth_buffer  # type: ignore[attr-defined]
        B, _C, Az, _Rg = rcmc_batch.shape
        Az_core = Az - 2 * az_buffer
        # Limit batches to n samples
        n = min(self.num_images, B)
        if n < B:
            rcmc_batch = rcmc_batch[:n]
            slc_batch = slc_batch[:n]
            metadata_list = metadata_list[:n]
            ephemeris_list = ephemeris_list[:n]
            coords_list = coords_list[:n]

        # ── Forward pass (no grad, on same device as model) ───────────
        with torch.no_grad():
            output = pl_module(rcmc_batch)
            x_hat_phys = minmax_inverse(output.x_hat, RC_MIN, RC_MAX)  # (B,2,Az+2*buf,Rg)

            # Azimuth-focus the denormalized reconstructed RCMC
            slc_recon = full_azimuth_compress_batch(
                x_hat_phys,
                metadata_list,
                ephemeris_list,
                buffer_size=az_buffer,
                device=str(output.x_hat.device),
                coords_batch=coords_list,
            )  # (B, 2, Az_core, Rg)  — physical SLC scale

        # ── Trim azimuth az_buffer; denormalize to physical scale ────────
        rcmc_input_core = minmax_inverse(
            rcmc_batch[:, :, az_buffer : az_buffer + Az_core, :], RC_MIN, RC_MAX
        )
        rcmc_recon_core = minmax_inverse(
            output.x_hat[:, :, az_buffer : az_buffer + Az_core, :], RC_MIN, RC_MAX
        )
        slc_target_core = minmax_inverse(
            slc_batch[:, :, az_buffer : az_buffer + Az_core, :], GT_MIN, GT_MAX
        )

        # ── Log-intensity Tensors on device (B, H, W) ─────────────────
        rows_data: list[Tensor] = [
            phys_to_logI_torch(rcmc_input_core),
            phys_to_logI_torch(rcmc_recon_core),
            phys_to_logI_torch(slc_recon),
            phys_to_logI_torch(slc_target_core),
        ]

        if self.verbose:
            print_images_statistics(
                {
                    title.split("\n")[0]: rows_data[i].cpu().numpy()
                    for i, title in enumerate(self.ROW_TITLES)
                },
                title=f"[MonitorValReconstruction] Epoch {trainer.current_epoch} — first {n} patches logI stats",
            )

        # ── Physical SLC min/max (F1: scale explosion diagnostics) ────────
        slc_r_min = float(slc_recon.abs().min().cpu())
        slc_r_max = float(slc_recon.abs().max().cpu())
        slc_t_min = float(slc_target_core.abs().min().cpu())
        slc_t_max = float(slc_target_core.abs().max().cpu())
        scale_info = (
            f"SLC recon |·| ∈ [{slc_r_min:.3e}, {slc_r_max:.3e}]   "
            f"SLC target |·| ∈ [{slc_t_min:.3e}, {slc_t_max:.3e}]"
        )

        # ── Metrics for per-patch titles and figure summary ───────────
        # Metrics are computed on normalised [0,1] tensors so they are
        # directly comparable to the module's validation_step scalars
        # (where forward_with_az_compression returns normalised SLC).
        # slc_batch is already normalised by the dataloader; slc_recon is
        # physical-scale and must be re-normalised before calling metrics.
        slc_recon_norm = minmax_normalize(slc_recon, GT_MIN, GT_MAX)
        slc_target_core_norm = slc_batch[:, :, az_buffer : az_buffer + Az_core, :]
        patch_metrics: list[tuple[float, float, float]] = []
        for col_idx in range(n):
            pred_patch = slc_recon_norm[col_idx : col_idx + 1]
            target_patch = slc_target_core_norm[col_idx : col_idx + 1]
            patch_psnr = psnr_amplitude(pred_patch, target_patch).item()
            patch_ssim = ssim_amplitude(pred_patch, target_patch).item()
            patch_coherence, _ = complex_correlation_metric(pred_patch, target_patch)
            patch_metrics.append((patch_psnr, patch_ssim, patch_coherence.item()))

        # ── Build figure ──────────────────────────────────────────────
        # Extra width per column to accommodate per-image colorbars
        fig, axes = plt.subplots(4, n, figsize=(5 * n, 17), squeeze=False)

        for row_idx, (row_title, row_tensor) in enumerate(zip(self.ROW_TITLES, rows_data)):
            row_imgs_clipped = [
                clip_mean_std_numpy(row_tensor[col_idx].cpu().numpy(), self.clip_factor)
                for col_idx in range(n)
            ]

            for col_idx, img_np in enumerate(row_imgs_clipped):
                ax = axes[row_idx, col_idx]
                # img_np shape is (Az, Rg): rows = azimuth (vertical, top→bottom),
                # cols = range (horizontal, left→right). No transpose needed.
                im = ax.imshow(
                    img_np,
                    cmap="viridis",
                    aspect="auto",
                    origin="upper",
                    vmin=img_np.min(),
                    vmax=img_np.max(),
                )
                ax.set_facecolor("black")
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
                    patch_psnr, patch_ssim, patch_coherence = patch_metrics[col_idx]
                    ax.set_title(
                        (
                            f"Patch {col_idx}\n"
                            f"PSNR: {patch_psnr:.2f} dB | "
                            f"SSIM: {patch_ssim:.3f} | "
                            f"Coherence: {patch_coherence:.3f}"
                        ),
                        fontsize=9,
                    )
                # Per-image colorbar (steals space from this axes only)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)

        # ── Summary metrics for the figure title ──────────────────────
        psnr_slc = psnr_amplitude(slc_recon_norm, slc_target_core_norm).item()
        ssim_slc = ssim_amplitude(slc_recon_norm, slc_target_core_norm).item()
        phase_err = phase_preservation_metric(slc_recon_norm, slc_target_core_norm)[0].item()
        coherence_loss = complex_coherence_loss(slc_recon_norm, slc_target_core_norm).item()
        corr_metric_mean, corr_metric_std = complex_correlation_metric(
            slc_recon_norm, slc_target_core_norm
        )
        kde_loss = kde_histogram_loss(slc_recon_norm, slc_target_core_norm).item()

        fig.suptitle(
            f"Validation epoch {trainer.current_epoch} — "
            f"SLC PSNR: {psnr_slc:.4f}dB, SSIM: {ssim_slc:.4f}, Phase Error: {phase_err:.4f}, Coherence Loss: {coherence_loss:.4f}, Corr Metric: {corr_metric_mean:.4f}±{corr_metric_std:.4f}, KDE Loss: {kde_loss:.4f}\n"
            f"{scale_info}",
            fontsize=10,
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
                    "val_batch/reconstructions": wandb.Image(fig),
                    "val_batch/psnr_slc": psnr_slc,
                    "val_batch/ssim_slc": ssim_slc,
                    "val_batch/phase_error": phase_err,
                    "val_batch/coherence_loss": coherence_loss,
                    "val_batch/correlation_metric_mean": corr_metric_mean.item(),
                    "val_batch/correlation_metric_std": corr_metric_std.item(),
                    "val_batch/kde_loss": kde_loss,
                    # F6: x_hat value distribution — detects collapse or saturation
                    "val_batch/x_hat_histogram": wandb.Histogram(
                        output.x_hat.detach().cpu().numpy().ravel()
                    ),
                },
                step=trainer.global_step,
            )

        plt.close(fig)
