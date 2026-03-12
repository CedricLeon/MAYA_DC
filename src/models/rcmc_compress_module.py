from typing import Any, Dict, List, Optional, Tuple

import lightning
import torch
from compressai.models import CompressionModel
from maya4 import GT_MAX, GT_MIN, RC_MAX, RC_MIN, minmax_inverse, minmax_normalize
from torch import Tensor

from src.models.components.scale_hyperprior import ForwardOutput, Likelihoods
from src.utils.sarpyx_azimuth_compression import full_azimuth_compress_batch


class RCMCDCmodule(lightning.LightningModule):
    """Lightning Module to compress Range Cell Migration Corrected (RCMC) SAR data.

    Pipeline per training step::

        RCMC (B,2,Az+2*buf,Rg)
            → ScaleHyperprior (compress + reconstruct)
            → x_hat (B,2,Az+2*buf,Rg)
            → full_azimuth_compress_batch  (sarpyx CoarseRDA)
            → SLC_recon (B,2,Az,Rg)
            → loss(SLC_recon, SLC_target)
    """

    def __init__(
        self,
        net: CompressionModel,
        criterion: torch.nn.Module,
        net_optimizer: torch.optim.Optimizer,
        aux_optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        azimuth_buffer: int = 512,
        gradient_clip_norm: float = 1.0,
        compile: bool = False,
    ):
        """Initialise the Lightning Module.

        Args:
            net: ScaleHyperprior (or any CompressAI model).
            criterion: Loss module (e.g. SimpleMSELoss).
            net_optimizer: Main optimizer for network parameters.
            aux_optimizer: Auxiliary optimizer for entropy bottleneck quantiles.
            scheduler: LR scheduler (ReduceLROnPlateau recommended).
            azimuth_buffer: Buffer size in azimuth; must satisfy
                ``(patch_size[0] + 2 * azimuth_buffer) % 16 == 0``.
            gradient_clip_norm: Max gradient norm for clipping.
            compile: Compile the model with ``torch.compile``.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        self.net = net
        self.criterion = criterion

        # Two optimizers → manual optimization
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> ForwardOutput:
        """Default forward pass."""
        return self.net(x)

    # ------------------------------------------------------------------
    def _log_metrics(
        self,
        prefix: str,
        criterion_dict: Dict[str, Any],
        aux_loss: float,
    ) -> None:
        """Log training/validation/test metrics with appropriate prefixes and settings."""
        log_info = {f"{prefix}/{key}": value for key, value in criterion_dict.items()}
        log_info[f"{prefix}/aux"] = aux_loss

        on_step, on_epoch, prog_bar, sync_dist = None, None, False, True
        if prefix == "train":
            on_step, on_epoch, prog_bar, sync_dist = True, False, False, True
        elif prefix == "valid":
            on_step, on_epoch, prog_bar, sync_dist = False, True, True, True
        elif prefix == "test":
            on_step, on_epoch, prog_bar, sync_dist = False, True, False, True

        self.log_dict(
            log_info,
            sync_dist=sync_dist,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
        )

    # ------------------------------------------------------------------
    def _extract_from_batch(
        self, batch
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[List], Optional[List]]:
        """Unpack a batch and extract model inputs and targets.

        The datamodule yields::

            (rcmc, slc, metadata_list, ephemeris_list, coords_list)

        where ``rcmc`` and ``slc`` both have shape ``(B, 2, Az+2*buf, Rg)``.
        We keep the full RCMC (with buffer) as the compressor input and trim
        the buffer from SLC/RCMC to get the loss targets.

        Returns:
            rcmc_batch:     ``(B, 2, Az+2*buf, Rg)``  - full RCMC fed to compressor
            slc_target:     ``(B, 2, Az, Rg)``        - ground-truth SLC (trimmed)
            rcmc_target:    ``(B, 2, Az, Rg)``        - RCMC core (for RCMC-domain loss)
            metadata_list:  list of DataFrames or None
            ephemeris_list: list of DataFrames or None
        """
        # Support both old 2-tuple batches and the new 5-tuple format
        if len(batch) == 5:
            rcmc_batch, slc_batch, metadata_list, ephemeris_list, _coords = batch
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}, expected 2 or 5")

        buffer = self.hparams.azimuth_buffer  # type: ignore[attr-defined]

        _B, _C, Az, _Rg = rcmc_batch.shape
        Az_core = Az - 2 * buffer
        rcmc_target = rcmc_batch[:, :, buffer : buffer + Az_core, :]
        slc_target = slc_batch[:, :, buffer : buffer + Az_core, :]

        return rcmc_batch, slc_target, rcmc_target, metadata_list, ephemeris_list

    def forward_with_az_compression(self, batch: Tensor) -> Any:
        rcmc_input, slc_target, _rcmc_target, metadata_list, ephemeris_list = (
            self._extract_from_batch(batch)
        )

        # Forward pass through compression model
        output = self.forward(rcmc_input)

        # Denormalize the reconstructed RCMC (CoarseRDA cannot operate on normalized values).
        x_hat_denorm = minmax_inverse(output.x_hat, RC_MIN, RC_MAX)
        # Azimuth compression with sarpyx (CoarseRDA) to get reconstructed SLC
        slc_recon_denorm = full_azimuth_compress_batch(
            x_hat_denorm,
            metadata_list,
            ephemeris_list,
            buffer_size=self.hparams.azimuth_buffer,  # type: ignore[attr-defined]
            device=str(output.x_hat.device),
        )
        # renormalize the reconstructed SLC for loss computation (criterion may expect normalized inputs)
        slc_recon = minmax_normalize(slc_recon_denorm, GT_MIN, GT_MAX)

        # Loss: compare reconstructed SLC to ground-truth SLC
        criterion_input = ForwardOutput(x_hat=slc_recon, likelihoods=output.likelihoods)
        return self.criterion(criterion_input, slc_target)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """Full training step with manual optimization."""
        criterion = self.forward_with_az_compression(batch)

        # Main network backward
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1] if len(optimizers) > 1 else None

        self.manual_backward(criterion["loss"])

        # DEBUG: check for null/zero gradients after backward — remove once confirmed working
        null_grad, zero_grad = [], []
        for name, p in self.net.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    null_grad.append(name)
                elif p.grad.abs().max() == 0:
                    zero_grad.append(name)
        if null_grad:
            self.print(f"[grad-check] NULL grad ({len(null_grad)} params): {null_grad[:5]}")
        if zero_grad:
            self.print(f"[grad-check] ZERO grad ({len(zero_grad)} params): {zero_grad[:5]}")
        if not null_grad and not zero_grad:
            self.print("[grad-check] OK — all gradients non-null and non-zero")

        self.clip_gradients(
            net_optimizer,  # type: ignore[attr-defined]
            gradient_clip_val=self.hparams.gradient_clip_norm,  # type: ignore[attr-defined]
            gradient_clip_algorithm="norm",
        )
        net_optimizer.step()
        net_optimizer.zero_grad()

        # Auxiliary (entropy bottleneck) backward
        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        if aux_optimizer is not None:
            aux_optimizer.step()
            aux_optimizer.zero_grad()

        lr = net_optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)
        self._log_metrics("train", criterion, aux_loss.item())

    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        """Validation step also with azimuth compression."""
        criterion = self.forward_with_az_compression(batch)
        self._log_metrics("valid", criterion, self.net.aux_loss().item())

    # ------------------------------------------------------------------
    def on_test_epoch_start(self) -> None:
        """Update entropy bottleneck CDF tables before testing."""
        # Only useful if we perform real compression in testing, not the case now.
        self.net.update(force=True)

    def test_step(self, batch, batch_idx):
        """Test step with likelihoods evaluation."""
        # @TODO: transform this classic testing that uses likelihhods into a real testing that performs actual compression
        criterion = self.forward_with_az_compression(batch)
        self._log_metrics("test", criterion, self.net.aux_loss().item())

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        """Step ReduceLROnPlateau with the monitored validation metric.

        FIX 11: only one scheduler step, in the right place, with the metric.
        """
        sch = self.lr_schedulers()
        if sch is not None and isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val_loss = self.trainer.callback_metrics.get("valid/loss")
            if val_loss is not None:
                sch.step(val_loss)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        main_params = [
            p
            for name, p in self.net.named_parameters()
            if p.requires_grad and not name.endswith(".quantiles")
        ]
        aux_params = [
            p
            for name, p in self.net.named_parameters()
            if p.requires_grad and name.endswith(".quantiles")
        ]

        all_params = {p for _, p in self.net.named_parameters() if p.requires_grad}
        assert not set(main_params) & set(aux_params), "Parameter overlap between main and aux"
        assert set(main_params) | set(aux_params) == all_params, "Parameters not fully covered"

        net_optimizer = self.hparams.net_optimizer(params=main_params)  # type: ignore[attr-defined]
        aux_optimizer = self.hparams.aux_optimizer(params=aux_params)  # type: ignore[attr-defined]

        if getattr(self.hparams, "scheduler", None) is not None:
            scheduler = self.hparams.scheduler(optimizer=net_optimizer)  # type: ignore[attr-defined]
            return (
                {
                    "optimizer": net_optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "name": "net_lr"},
                },
                {"optimizer": aux_optimizer},
            )

        return net_optimizer, aux_optimizer
