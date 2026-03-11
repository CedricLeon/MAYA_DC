from __future__ import annotations

from typing import Any, Optional, Sequence

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.maya_sar_dataset import MAYASARPatchDataset, sar_batch_collate
from src.utils.sar_environment import ensure_sar_environment


class MAYASARDataModule(LightningDataModule):
    """Lightning data module for MAYA/Sarpyx patch-based SAR training."""

    def __init__(
        self,
        data_dir: str = "data/test_complete_download",
        train_partitions: Sequence[str] = ("PT1",),
        val_partitions: Sequence[str] = ("PT2",),
        test_partitions: Sequence[str] = ("PT4",),
        patch_size: tuple[int, int] = (512, 512),
        stride: tuple[int, int] | None = None,
        azimuth_margin: int = 500,
        range_margin: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        level_from: str = "rcmc",
        level_to: str = "az",
        max_products: int | None = None,
        enforce_full_context: bool = True,
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        ensure_sar_environment()

    def _build_dataset(self, partitions: Sequence[str]) -> MAYASARPatchDataset:
        return MAYASARPatchDataset(
            data_dir=self.hparams.data_dir,
            partitions=partitions,
            patch_size=tuple(self.hparams.patch_size),
            stride=None if self.hparams.stride is None else tuple(self.hparams.stride),
            azimuth_margin=self.hparams.azimuth_margin,
            range_margin=self.hparams.range_margin,
            level_from=self.hparams.level_from,
            level_to=self.hparams.level_to,
            max_products=self.hparams.max_products,
            enforce_full_context=self.hparams.enforce_full_context,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.batch_size != 1:
            raise RuntimeError(
                "MAYASARDataModule currently requires batch_size=1 because focus metadata is "
                "built per sample."
            )

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if stage in (None, "fit") and self.data_train is None:
            self.data_train = self._build_dataset(self.hparams.train_partitions)
            self.data_val = self._build_dataset(self.hparams.val_partitions)

        if stage in (None, "test") and self.data_test is None:
            self.data_test = self._build_dataset(self.hparams.test_partitions)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            shuffle=self.hparams.shuffle_train,
            collate_fn=sar_batch_collate,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            shuffle=False,
            collate_fn=sar_batch_collate,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            shuffle=False,
            collate_fn=sar_batch_collate,
        )
