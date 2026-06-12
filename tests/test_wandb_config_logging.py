from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import template_utils


class _CaptureLogger:
    def __init__(self) -> None:
        self.hparams = None

    def log_hyperparams(self, params):
        self.hparams = params


class _CaptureTrainer:
    def __init__(self, logger: _CaptureLogger) -> None:
        self.logger = logger
        self.loggers = [logger]


def _build_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "_target_": "src.models.demo.DemoModule",
                "training_mode": "slc",
                "net": {
                    "_target_": "src.models.components.FactorizedPrior",
                    "activation": "gdn",
                },
                "criterion": {
                    "_target_": "src.models.components.losses.SimpleMSE",
                    "lmbda": 1.0,
                },
                "optimizer": {"lr": 1e-3},
            },
            "data": {
                "batch_size": 4,
                "loader": {"num_workers": 2, "pin_memory": False},
                "stats": {"train": [0.1, 0.2], "val": [0.3, 0.4]},
            },
            "trainer": {"max_epochs": 1},
            "callbacks": {"example": {"enabled": True}},
            "logger": {
                "wandb": {
                    "entity": "entity",
                    "project": "project",
                    "save_dir": str(tmp_path),
                    "offline": False,
                }
            },
            "tags": ["test"],
            "task_name": "wandb-config-test",
            "seed": 7,
            "azimuth_buffer": 16,
        }
    )


def test_early_wandb_initialization_keeps_nested_dicts(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_cfg(tmp_path)
    captured = {}

    def _fake_init(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(template_utils.wandb, "init", _fake_init)

    template_utils.early_wandb_initialization(cfg)

    assert isinstance(captured["config"]["data"], dict)
    assert captured["config"]["data"]["loader"]["num_workers"] == 2
    assert captured["config"]["data"]["stats"]["train"] == [0.1, 0.2]


def test_log_hyperparameters_keeps_nested_dicts(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    logger = _CaptureLogger()
    trainer = _CaptureTrainer(logger)
    model = torch.nn.Linear(3, 2)

    template_utils.log_hyperparameters(
        {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
        }
    )

    assert isinstance(logger.hparams["data"], dict)
    assert logger.hparams["data"]["loader"]["pin_memory"] is False
    assert isinstance(logger.hparams["model"], dict)
    assert logger.hparams["model"]["criterion"]["lmbda"] == 1.0
