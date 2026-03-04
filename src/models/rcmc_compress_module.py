from typing import Any, Dict, Optional, Tuple

import lightning
import torch
import torchmetrics.functional.image as F
from compressai.models import CompressionModel
from torch import Tensor


class RCMCDCmodule(lightning.LightningModule):
    """Lightning Module to compress Range Cell Migration Corrected (RCMC) SAR data."""

    def __init__(
        self,
        net: CompressionModel,
        criterion: torch.nn.Module,
        net_optimizer: torch.optim.Optimizer,
        aux_optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        gradient_clip_norm: float = 1.0,
        compile: bool = False,
    ):
        """Initialize the Lightning Module.

        Args:
            net: Neural network module
            criterion: Loss criterion
            net_optimizer: Main optimizer for network parameters
            aux_optimizer: Auxiliary optimizer for quantiles
            scheduler: Learning rate scheduler
            gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
            compile: Whether to compile the model (default: False)
        """
        super().__init__()

        # Save hyperparameters to be accessible via self.hparams (ignore nn.Modules)
        self.save_hyperparameters(ignore=["criterion", "net"], logger=False)

        # Hydra recursive instantiation.
        self.net = net
        self.criterion = criterion

        # Activate manual optimization, because we have two optimizers.
        self.automatic_optimization = False

    def forward(self, x: Tensor):
        """Forward pass through the network."""
        return self.net(x)

    def _log_metrics(
        self,
        prefix: str,
        criterion: Dict[str, Any],
        aux_loss: float,
    ) -> None:
        """Log training, validation, or test metrics."""
        log_info = {f"{prefix}/{key}": value for key, value in criterion.items()}
        log_info[f"{prefix}/aux"] = aux_loss

        # Configure per prefix (e.g. train/valid/test) logging **kwargs.
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

    def some_function(self, batch) -> Tuple[Tensor, Tensor]:
        """Extract inputs and targets from the batch.

        Placeholder for actual logic.
        """
        x_batch, y_batch = batch
        return x_batch, y_batch

    def training_step(self, batch, batch_idx):
        """Training step where we manually optimize because we have 2 optimizers."""
        net_optimizer, aux_optimizer = self.optimizers()

        # Forward pass
        rcmc_input, slc_target = self.some_function(batch)
        rcmc_recon = self.forward(rcmc_input)
        criterion = self.criterion(rcmc_recon, slc_target)

        # Backward pass for the main loss
        self.manual_backward(criterion["loss"])
        self.clip_gradients(
            net_optimizer,  # type: ignore[attr-defined]
            gradient_clip_val=self.hparams.gradient_clip_norm,  # type: ignore[attr-defined]
            gradient_clip_algorithm="norm",
        )
        net_optimizer.step()
        net_optimizer.zero_grad()

        # Auxiliary loss (entropy bottleneck) if available
        aux_loss = self.net.aux_loss()
        self.manual_backward(aux_loss)
        aux_optimizer.step()
        aux_optimizer.zero_grad()

        # Step scheduler if available (for epoch-based schedulers)
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()  # type: ignore[attr-defined]

        # custom learning rate logging
        lr = net_optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # Log metrics
        self._log_metrics("train", criterion, aux_loss.item())

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        rcmc_input, slc_target = self.some_function(batch)
        rcmc_recon = self.forward(rcmc_input)
        criterion = self.criterion(rcmc_recon, slc_target)
        aux_loss = self.net.aux_loss()
        self._log_metrics("valid", criterion, aux_loss.item())

    def on_test_epoch_start(self) -> None:
        """Update the entropy bottleneck tables before testing."""
        self.net.update(force=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        # # Log extra metrics
        # self.log_dict(
        #     all_metrics,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        # )
        pass

    def on_validation_epoch_end(self) -> None:
        """Update LR scheduler based on validation loss."""
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(self.trainer.callback_metrics["valid/loss"])

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar, you might have multiple.

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        main_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        ]
        aux_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        ]

        # Validation: Ensure no parameter overlap and all parameters are accounted for
        all_params = {param for _, param in self.net.named_parameters() if param.requires_grad}
        assert not set(main_params) & set(
            aux_params
        )  # "Intersection found in main and auxiliary parameters"
        assert (
            set(main_params) | set(aux_params) == all_params
        )  # "Union of main and auxiliary parameters does not match all model parameters"

        # Instantiate optimizer(s)
        net_optimizer = self.hparams.net_optimizer(params=main_params)  # type: ignore[attr-defined]
        aux_optimizer = self.hparams.aux_optimizer(params=aux_params)  # type: ignore[attr-defined]

        if getattr(self.hparams, "scheduler", None) is not None:
            scheduler = self.hparams.scheduler(optimizer=net_optimizer)  # type: ignore[attr-defined]
            return (
                {
                    "optimizer": net_optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "name": "net_lr",
                    },
                },
                {"optimizer": aux_optimizer},
            )

        return net_optimizer, aux_optimizer
