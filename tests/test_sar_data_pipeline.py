from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from src.data.components.maya_sar_dataset import MAYASARPatchDataset, sar_batch_collate
from src.models.components.azimuth_focus import TorchAzimuthFocus
from src.models.criteria import SARRateDistortionCriterion
from src.models.rcmc_compress_module import RCMCDCmodule


class DummyCompressionNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
        self.latents = torch.nn.Parameter(torch.zeros(1))
        self.quantiles = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = self.proj(x)
        likelihood = torch.sigmoid(self.latents).expand_as(x_hat[:, :1, ...])
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": likelihood,
                "z": torch.sigmoid(self.quantiles).view(1, 1, 1, 1).expand_as(likelihood),
            },
        }

    def aux_loss(self) -> torch.Tensor:
        return self.quantiles.square().sum()


class SyntheticSARDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.focus = TorchAzimuthFocus()
        self.sample = {
            "rcmc_input": torch.randn(2, 6, 4),
            "slc_target": torch.randn(2, 4, 4),
            "focus_metadata": {
                "slant_range_vec": torch.linspace(800.0, 900.0, 4, dtype=torch.float64),
                "azimuth_freq_hz": torch.linspace(-250.0, 250.0, 6, dtype=torch.float64),
                "effective_velocity": torch.linspace(7200.0, 7300.0, 6, dtype=torch.float64),
                "wavelength_m": 0.055465764662349676,
                "crop_azimuth": (1, 1),
                "crop_range": (0, 0),
            },
        }

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        sample = {
            "rcmc_input": self.sample["rcmc_input"].clone(),
            "focus_metadata": self.sample["focus_metadata"],
        }
        focused_target = self.focus(sample["rcmc_input"].unsqueeze(0), sample["focus_metadata"]).squeeze(0)
        sample["slc_target"] = torch.stack((focused_target.real, focused_target.imag), dim=0)
        return sample


def test_maya_sar_dataset_emits_expected_batch_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product_dir = tmp_path / "test_complete_download" / "PT1" / "sample.zarr"
    product_dir.mkdir(parents=True)

    arrays = {
        "rcmc": (np_real := torch.randn(8, 6).numpy()) + 1j * torch.randn(8, 6).numpy(),
        "az": torch.randn(8, 6).numpy() + 1j * torch.randn(8, 6).numpy(),
    }

    class FakeProductHandler:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_array(self, name: str):
            return arrays[name]

    monkeypatch.setattr(
        "src.data.components.maya_sar_dataset.ensure_sar_environment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.data.components.maya_sar_dataset._open_product_handler",
        lambda path: FakeProductHandler(path),
    )
    monkeypatch.setattr(
        "src.data.components.maya_sar_dataset._load_product_layout",
        lambda *args, **kwargs: (8, (0, 8), (0, 6)),
    )
    monkeypatch.setattr(
        "src.data.components.maya_sar_dataset._build_focus_metadata",
        lambda **kwargs: {
            "slant_range_vec": torch.linspace(800.0, 900.0, 3, dtype=torch.float64),
            "azimuth_freq_hz": torch.linspace(-250.0, 250.0, 6, dtype=torch.float64),
            "effective_velocity": torch.linspace(7200.0, 7300.0, 6, dtype=torch.float64),
            "wavelength_m": 0.055465764662349676,
            "crop_azimuth": kwargs["crop_azimuth"],
            "crop_range": kwargs["crop_range"],
        },
    )

    dataset = MAYASARPatchDataset(
        data_dir=tmp_path / "test_complete_download",
        partitions=("PT1",),
        patch_size=(4, 3),
        stride=(4, 3),
        azimuth_margin=1,
        range_margin=0,
    )

    sample = dataset[0]

    assert sample["rcmc_input"].shape == (2, 6, 3)
    assert sample["slc_target"].shape == (2, 4, 3)
    assert sample["focus_metadata"]["crop_azimuth"] == (1, 1)
    assert sample["focus_metadata"]["crop_range"] == (0, 0)


def test_sar_batch_collate_requires_single_sample() -> None:
    sample = {
        "rcmc_input": torch.randn(2, 6, 4),
        "slc_target": torch.randn(2, 4, 4),
        "focus_metadata": {"crop_azimuth": (1, 1), "crop_range": (0, 0)},
    }
    batch = sar_batch_collate([sample])

    assert batch["rcmc_input"].shape == (1, 2, 6, 4)
    assert batch["slc_target"].shape == (1, 2, 4, 4)

    with pytest.raises(RuntimeError):
        sar_batch_collate([sample, sample])


def test_rcmc_module_trains_through_lightning_trainer() -> None:
    dataset = SyntheticSARDataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=sar_batch_collate)

    model = RCMCDCmodule(
        net=DummyCompressionNet(),
        criterion=SARRateDistortionCriterion(lmbda=0.2),
        focus=TorchAzimuthFocus(),
        net_optimizer=partial(torch.optim.SGD, lr=0.1),
        aux_optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
    )
    before = model.net.proj.weight.detach().clone()

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    after = model.net.proj.weight.detach()
    assert not torch.allclose(before, after)
