from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import lightning
import torch
from torch import Tensor, nn

from src.models.components.azimuth_focus import IdentityFocus

try:
    from compressai.models import CompressionModel
except ModuleNotFoundError:  # pragma: no cover - optional dependency at import time
    CompressionModel = nn.Module  # type: ignore[misc,assignment]


class RCMCDCmodule(lightning.LightningModule):
    """Lightning module for differentiable RCMC-to-SLC compression training."""

    def __init__(
        self,
        net: CompressionModel,
        criterion: nn.Module,
        net_optimizer: torch.optim.Optimizer,
        aux_optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        focus: Optional[nn.Module] = None,
        gradient_clip_norm: float = 1.0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion", "focus", "net"], logger=False)

        self.net = net
        self.criterion = criterion
        self.focus = focus if focus is not None else IdentityFocus()
        self.automatic_optimization = False

        if compile:
            self.net = torch.compile(self.net)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        output = self.net(x)
        if not isinstance(output, Mapping):
            raise TypeError(f"Compression model must return a mapping, got {type(output)!r}")
        if "x_hat" not in output:
            raise KeyError("Compression model output must contain an 'x_hat' tensor")
        return dict(output)

    def _log_metrics(
        self,
        prefix: str,
        criterion: Mapping[str, Tensor],
        aux_loss: Tensor,
    ) -> None:
        log_info = {f"{prefix}/{key}": value for key, value in criterion.items()}
        log_info[f"{prefix}/aux"] = aux_loss

        on_step, on_epoch, prog_bar = False, True, False
        if prefix == "train":
            on_step, on_epoch = True, False
        elif prefix == "valid":
            prog_bar = True

        self.log_dict(
            log_info,
            sync_dist=self.trainer is not None and self.trainer.world_size > 1,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
        )

    @staticmethod
    def _as_batch_mapping(batch: Any) -> Mapping[str, Any]:
        if isinstance(batch, Mapping):
            return batch

        if isinstance(batch, Sequence):
            if len(batch) == 2:
                return {"rcmc_input": batch[0], "slc_target": batch[1], "focus_metadata": None}
            if len(batch) == 3:
                return {
                    "rcmc_input": batch[0],
                    "slc_target": batch[1],
                    "focus_metadata": batch[2],
                }

        raise TypeError(
            "Batch must be a mapping or a tuple/list shaped as "
            "(rcmc_input, slc_target[, focus_metadata])"
        )

    def _unpack_batch(self, batch: Any) -> tuple[Tensor, Tensor, Any]:
        batch_map = self._as_batch_mapping(batch)

        if "rcmc_input" not in batch_map or "slc_target" not in batch_map:
            raise KeyError("Batch must contain 'rcmc_input' and 'slc_target'")

        return batch_map["rcmc_input"], batch_map["slc_target"], batch_map.get("focus_metadata")

    def model_step(self, batch: Any) -> tuple[dict[str, Tensor], Tensor]:
        rcmc_input, slc_target, focus_metadata = self._unpack_batch(batch)
        model_output = self.forward(rcmc_input)
        focused_prediction = self.focus(model_output["x_hat"], focus_metadata)

        criterion = self.criterion(
            prediction=focused_prediction,
            target=slc_target,
            likelihoods=model_output.get("likelihoods"),
        )
        if not isinstance(criterion, Mapping) or "loss" not in criterion:
            raise TypeError("Criterion must return a mapping containing a differentiable 'loss'")

        loss = criterion["loss"]
        if not isinstance(loss, Tensor):
            raise TypeError("Criterion 'loss' must be a tensor")
        if self.training and not loss.requires_grad:
            raise RuntimeError(
                "Training loss does not require gradients. Check the prediction path for "
                "NumPy conversions or detached tensors."
            )

        aux_loss_fn = getattr(self.net, "aux_loss", None)
        if not callable(aux_loss_fn):
            aux_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        else:
            aux_loss = aux_loss_fn()
            if not isinstance(aux_loss, Tensor):
                raise TypeError("Compression model auxiliary loss must be a tensor")

        return dict(criterion), aux_loss

    def training_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        net_optimizer, aux_optimizer = self.optimizers()
        net_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        criterion, aux_loss = self.model_step(batch)

        self.manual_backward(criterion["loss"])
        self.clip_gradients(
            net_optimizer,  # type: ignore[attr-defined]
            gradient_clip_val=self.hparams.gradient_clip_norm,  # type: ignore[attr-defined]
            gradient_clip_algorithm="norm",
        )
        net_optimizer.step()

        if aux_loss.requires_grad:
            self.manual_backward(aux_loss)
            aux_optimizer.step()

        scheduler = self.lr_schedulers()
        if self.trainer.is_last_batch:
            if scheduler is None:
                pass
            elif isinstance(scheduler, (list, tuple)):
                for item in scheduler:
                    if item is not None and not isinstance(
                        item, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        item.step()
            elif not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        self.log(
            "train/lr",
            net_optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self._log_metrics("train", criterion, aux_loss.detach())

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        criterion, aux_loss = self.model_step(batch)
        self._log_metrics("valid", criterion, aux_loss.detach())

    def on_test_epoch_start(self) -> None:
        update = getattr(self.net, "update", None)
        if callable(update):
            update(force=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        criterion, aux_loss = self.model_step(batch)
        self._log_metrics("test", criterion, aux_loss.detach())

    def on_validation_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid/loss"])
        elif isinstance(scheduler, (list, tuple)):
            for item in scheduler:
                if isinstance(item, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    item.step(self.trainer.callback_metrics["valid/loss"])

    def configure_optimizers(self):
        def is_aux_parameter(name: str) -> bool:
            return name == "quantiles" or name.endswith(".quantiles")

        main_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and not is_aux_parameter(name)
        ]
        aux_params = [
            param
            for name, param in self.net.named_parameters()
            if param.requires_grad and is_aux_parameter(name)
        ]

        all_params = {param for _, param in self.net.named_parameters() if param.requires_grad}
        assert not set(main_params) & set(aux_params)
        assert set(main_params) | set(aux_params) == all_params

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
