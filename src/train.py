import os
import socket
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning
import rootutils
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    early_wandb_initialization,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _uses_multiple_devices(devices: Any) -> bool:
    """Return whether the trainer config requests more than one device."""
    if isinstance(devices, int):
        return devices != 1
    if isinstance(devices, str):
        return devices != "1"
    if isinstance(devices, (list, tuple)):
        return len(devices) != 1
    return True


def _find_free_local_port() -> str:
    """Reserve an ephemeral localhost port for DDP bootstrap."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return str(sock.getsockname()[1])


def _configure_single_node_ddp_environment(cfg: DictConfig) -> None:
    """Fill in safe defaults for single-node GPU DDP when the scheduler didn't."""
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        return

    strategy = str(trainer_cfg.get("strategy", "")).lower()
    accelerator = str(trainer_cfg.get("accelerator", "")).lower()
    num_nodes = int(trainer_cfg.get("num_nodes", 1))
    devices = trainer_cfg.get("devices", 1)

    if "ddp" not in strategy:
        return
    if accelerator not in {"gpu", "cuda"}:
        return
    if num_nodes != 1 or not _uses_multiple_devices(devices):
        return

    previous_nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME")
    previous_gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME")

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = _find_free_local_port()

    # On single-node jobs, DDP only needs a local bootstrap interface.
    # Force loopback here because cluster-wide NCCL socket vars can point at
    # interfaces that are unavailable or unrouted inside the batch job.
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    log.info(
        "Configured single-node DDP bootstrap on loopback "
        f"<MASTER_ADDR={os.environ['MASTER_ADDR']}, "
        f"MASTER_PORT={os.environ['MASTER_PORT']}, "
        f"NCCL_SOCKET_IFNAME={os.environ['NCCL_SOCKET_IFNAME']}, "
        f"GLOO_SOCKET_IFNAME={os.environ['GLOO_SOCKET_IFNAME']}>"
    )
    if previous_nccl_ifname and previous_nccl_ifname != "lo":
        log.info(
            f"Overrode inherited NCCL_SOCKET_IFNAME <{previous_nccl_ifname}> for single-node DDP"
        )
    if previous_gloo_ifname and previous_gloo_ifname != "lo":
        log.info(
            f"Overrode inherited GLOO_SOCKET_IFNAME <{previous_gloo_ifname}> for single-node DDP"
        )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)
    if cfg.get("determinism"):
        # There is a weird incompatibility between cloudpickle and cudnn; TypeError: cannot pickle '_Deterministic' object
        # You can read more about it here: https://github.com/pytorch/pytorch/issues/48832 and https://github.com/cloudpipe/cloudpickle/issues/405
        # See https://github.com/ray-project/ray/issues/8569, for different fixes
        # One solution is to import torch in the train function
        import torch

        log.info("Setting deterministic behavior!")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _configure_single_node_ddp_environment(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.get("ckpt_path")
        if cfg.get("train"):
            checkpoint_callback = trainer.checkpoint_callback
            if (
                isinstance(checkpoint_callback, ModelCheckpoint)
                and checkpoint_callback.best_model_path
            ):
                ckpt_path = checkpoint_callback.best_model_path
            else:
                log.warning(
                    "Best ckpt not found / ModelCheckpoint not configured! "
                    "Using current weights for testing..."
                )
                ckpt_path = None
        elif ckpt_path in {"", None}:
            log.warning(
                "No ckpt_path provided for test-only run! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    wandb_on = (
        cfg.get("debug") is None and OmegaConf.select(cfg, "logger.wandb._target_") is not None
    )
    # Manual and early initialization of the W&B Run if no debug is planned
    if wandb_on:
        early_wandb_initialization(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # Even if it is managed automatically, manually call wandb.finish()
    # See: https://github.com/wandb/wandb/issues/6952, sometimes, offline runs seem to not upload the config or summary on W&B
    if wandb_on:
        wandb.finish()

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
