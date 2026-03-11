from typing import Any, Dict, List, Optional, Tuple

import lightning
import torch
from compressai.models import CompressionModel
from maya4 import RC_MAX, RC_MIN, minmax_inverse
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
        return self.net(x)

    # ------------------------------------------------------------------
    def _log_metrics(
        self,
        prefix: str,
        criterion_dict: Dict[str, Any],
        aux_loss: float,
    ) -> None:
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
    def _extract_inputs_targets(
        self, batch
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[List], Optional[List]]:
        """Unpack a batch and extract model inputs and targets.

        The datamodule yields::

            (rcmc, slc, metadata_list, ephemeris_list, coords_list)

        where ``rcmc`` and ``slc`` both have shape ``(B, 2, Az+2*buf, Rg)``.
        We keep the full RCMC (with buffer) as the compressor input and trim
        the buffer from SLC/RCMC to get the loss targets.

        Returns:
            rcmc_input:     ``(B, 2, Az+2*buf, Rg)``  - full RCMC fed to compressor
            slc_target:     ``(B, 2, Az, Rg)``        - ground-truth SLC (trimmed)
            rcmc_target:    ``(B, 2, Az, Rg)``        - RCMC core (for RCMC-domain loss)
            metadata_list:  list of DataFrames or None
            ephemeris_list: list of DataFrames or None
        """
        # Support both old 2-tuple batches and the new 5-tuple format
        if len(batch) == 5:
            rcmc_batch, slc_batch, metadata_list, ephemeris_list, _coords = batch
        elif len(batch) == 2:
            rcmc_batch, slc_batch = batch
            metadata_list, ephemeris_list = None, None
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}, expected 2 or 5")

        buffer = self.hparams.azimuth_buffer  # type: ignore[attr-defined]

        if rcmc_batch.dim() == 4:  # (B, C, Az, Rg)
            B, C, Az, Rg = rcmc_batch.shape
            Az_core = Az - 2 * buffer
            rcmc_target = rcmc_batch[:, :, buffer : buffer + Az_core, :]
            slc_target = slc_batch[:, :, buffer : buffer + Az_core, :]
        else:
            # Unexpected shape - fall back to no trimming
            rcmc_target = rcmc_batch
            slc_target = slc_batch

        return rcmc_batch, slc_target, rcmc_target, metadata_list, ephemeris_list

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        net_optimizer = optimizers[0]
        aux_optimizer = optimizers[1] if len(optimizers) > 1 else None

        rcmc_input, slc_target, _rcmc_target, metadata_list, ephemeris_list = (
            self._extract_inputs_targets(batch)
        )

        # Forward pass through compression model
        output = self.forward(rcmc_input)

        # Azimuth compression on reconstructed RCMC (with buffer)
        # Denormalize x_hat from [0,1] back to physical IQ scale before focusing;
        # CoarseRDA must operate on real-amplitude data, not normalized values.
        x_hat_phys = minmax_inverse(output.x_hat.detach(), RC_MIN, RC_MAX)
        slc_recon = full_azimuth_compress_batch(
            x_hat_phys,
            metadata_list,
            ephemeris_list,
            buffer_size=self.hparams.azimuth_buffer,  # type: ignore[attr-defined]
            device=str(output.x_hat.device),
        )

        # Loss: compare reconstructed SLC to ground-truth SLC
        criterion_input = ForwardOutput(x_hat=slc_recon, likelihoods=output.likelihoods)
        criterion = self.criterion(criterion_input, slc_target)

        # Main network backward
        self.manual_backward(criterion["loss"])
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

        # FIX 11: do NOT call sch.step() here; ReduceLROnPlateau is stepped
        # in on_validation_epoch_end with the monitored metric.

        lr = net_optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)
        self._log_metrics("train", criterion, aux_loss.item())

    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        rcmc_input, slc_target, _rcmc_target, metadata_list, ephemeris_list = (
            self._extract_inputs_targets(batch)
        )

        output = self.forward(rcmc_input)
        x_hat_phys = minmax_inverse(output.x_hat.detach(), RC_MIN, RC_MAX)
        slc_recon = full_azimuth_compress_batch(
            x_hat_phys,
            metadata_list,
            ephemeris_list,
            buffer_size=self.hparams.azimuth_buffer,  # type: ignore[attr-defined]
            device=str(output.x_hat.device),
        )

        criterion_input = ForwardOutput(x_hat=slc_recon, likelihoods=output.likelihoods)
        criterion = self.criterion(criterion_input, slc_target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("valid", criterion, aux_loss.item())

    # ------------------------------------------------------------------
    def on_test_epoch_start(self) -> None:
        """Update entropy bottleneck CDF tables before testing."""
        # Only useful if we perform real compression in testing, not the case now.
        self.net.update(force=True)

    def test_step(self, batch, batch_idx):
        rcmc_input, slc_target, _rcmc_target, metadata_list, ephemeris_list = (
            self._extract_inputs_targets(batch)
        )

        output = self.forward(rcmc_input)
        x_hat_phys = minmax_inverse(output.x_hat.detach(), RC_MIN, RC_MAX)
        slc_recon = full_azimuth_compress_batch(
            x_hat_phys,
            metadata_list,
            ephemeris_list,
            buffer_size=self.hparams.azimuth_buffer,  # type: ignore[attr-defined]
            device=str(output.x_hat.device),
        )

        criterion_input = ForwardOutput(x_hat=slc_recon, likelihoods=output.likelihoods)
        criterion = self.criterion(criterion_input, slc_target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("test", criterion, aux_loss.item())

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
