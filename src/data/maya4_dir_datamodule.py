"""MAYA4 SAR DataModule — directory-based product selection.

Directory layout expected::

    <train_dir>/
        [PT*/]            ← MAYA4's natural part-folder layout (optional)
            s1a-s*-raw-s-*.zarr
    <val_dir>/ ...
    <test_dir>/ ...

All ``*.zarr`` products found recursively in each directory are eligible,
then filtered by the ephemeris check and capped by ``max_products_*``.
``max_products_*=-1`` (default) means use all products that pass the check.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import lightning
import pandas as pd
import rootutils
from torch.utils.data import DataLoader

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from maya4 import (
    GT_MAX,
    GT_MIN,
    RC_MAX,
    RC_MIN,
    KPatchSampler,
    SARTransform,
    parse_product_filename,
)
from maya4.normalization import NormalizationModule

from src.data.maya4_datamodule import RCMCSARDataset, _collate_sar_batch

log = logging.getLogger(__name__)


def _parse_product_filename_compat(filename: str | Path) -> dict | None:
    """Parse both legacy `s1a-...` and newer `s1c-...` MAYA product names."""
    parsed = parse_product_filename(filename)
    if parsed is not None:
        return parsed

    sep = re.escape(os.sep)
    pattern = (
        rf"(?P<part>[a-zA-Z0-9]+){sep}(?P<mission>s1[a-zA-Z0-9]+)-s(?P<stripmap_mode>[a-zA-Z0-9]+)-raw-s-"
        r"(?P<polarization>[a-zA-Z0-9]+)-(?P<start_date>\d{8})t\d+-\d{8}t\d+-\d+-[a-zA-Z0-9]+\.zarr"
    )
    filename = Path(filename)
    parent_and_name = str(Path(filename.parent.name) / filename.name)
    match = re.match(pattern, parent_and_name)
    if not match:
        return None

    stripmap_mode = match.group("stripmap_mode")
    polarization = match.group("polarization")
    start_date = match.group("start_date")
    product_name = re.sub(r"-s\d+-raw-s-\w+-", "-", filename.name).split(".zarr")[0]

    return {
        "product_name": product_name,
        "stripmap_mode": int(stripmap_mode),
        "polarization": polarization,
        "acquisition_date": datetime.strptime(start_date, "%Y%m%d"),
        "full_name": filename,
        "part": match.group("part"),
        "store": None,
        "lat": None,
        "lon": None,
        "samples": [],
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RCMCSARDirDataset(RCMCSARDataset):
    """RCMCSARDataset variant that loads all products from a single directory.

    Overrides ``_build_file_list`` to scan ``product_dir`` recursively instead
    of running the full ``data_dir`` rglob + SampleFilter pipeline.  Everything
    else (store opening, ephemeris filtering, patch sampling) is unchanged.

    ``_product_dir`` must be set before ``super().__init__()`` because
    ``_build_file_list`` is invoked inside ``SARZarrDataset.__init__``.
    """

    def __init__(
        self,
        product_dir: str | Path,
        azimuth_buffer: int,
        max_products: int | None = None,
        **kwargs,
    ):
        self._product_dir = Path(product_dir)
        super().__init__(azimuth_buffer=azimuth_buffer, max_products=max_products, **kwargs)

    def _build_file_list(self) -> None:
        """Scans ``self._product_dir`` recursively for ``*.zarr`` files, parses their filenames,"""
        found = sorted(self._product_dir.rglob("*.zarr"))
        records = [r for f in found if (r := _parse_product_filename_compat(f)) is not None]
        if skipped := len(found) - len(records):
            log.warning("RCMCSARDirDataset: %d zarr paths could not be parsed, skipped.", skipped)
        self._files = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(
                columns=[
                    "full_name",
                    "part",
                    "stripmap_mode",
                    "polarization",
                    "acquisition_date",
                    "store",
                    "lat",
                    "lon",
                    "samples",
                ]
            )
        )
        self._files.sort_values("full_name", inplace=True, ignore_index=True)
        log.info("RCMCSARDirDataset: %d products in '%s'.", len(self._files), self._product_dir)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class MAYA4DirDataModule(lightning.LightningDataModule):
    """`LightningDataModule` for MAYA4 SAR data with directory-based product selection.

    Each batch is a 5-tuple::

        (rcmc, slc, metadata_list, ephemeris_list, coords_list)

    * ``rcmc``           - ``(B, 2, Az+2*buf, Rg)`` float32, normalised to [0, 1]
    * ``slc``            - ``(B, 2, Az+2*buf, Rg)`` float32, normalised to [0, 1]
    * ``metadata_list``  - one DataFrame per item, azimuth-sliced + SWST-corrected
    * ``ephemeris_list`` - one DataFrame per item
    * ``coords_list``    - one dict per item: ``{"zfile", "y", "x"}``
    """

    def __init__(
        self,
        train_dir: str | None = None,
        val_dir: str | None = None,
        test_dir: str | None = None,
        patch_size: tuple = (512, 512),
        azimuth_buffer: int = 512,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_products_train: int = -1,
        max_products_val: int = -1,
        max_products_test: int = -1,
        samples_per_prod: int = 100,
        online: bool = False,
    ) -> None:
        """
        Args:
            train_dir / val_dir / test_dir: Directories containing zarr products for
                each split.  All ``*.zarr`` files found recursively are eligible.
                ``None`` produces an empty DataLoader for that split.
            patch_size: Core patch ``(azimuth_lines, range_samples)``.
            azimuth_buffer: Extra lines on each side.
                ``patch_size[0] + 2 * azimuth_buffer`` must be divisible by 16.
            batch_size: Samples per batch (split across GPUs in DDP).
            num_workers / pin_memory: DataLoader settings.
            max_products_train/val/test: Cap on products per split after the
                ephemeris check.  ``-1`` means use all available products.
            samples_per_prod: Patches drawn per product per epoch.  ``0`` = all.
            online: Download missing zarr chunks from HuggingFace on demand.
        """
        super().__init__()
        total_az = patch_size[0] + 2 * azimuth_buffer
        assert total_az % 16 == 0, (
            f"patch_size[0] ({patch_size[0]}) + 2*azimuth_buffer ({2 * azimuth_buffer}) "
            f"= {total_az} is not divisible by 16."
        )
        self.save_hyperparameters(logger=False)
        self.transforms = SARTransform(
            transform_rcmc=NormalizationModule(data_min=RC_MIN, data_max=RC_MAX),
            transform_az=NormalizationModule(data_min=GT_MIN, data_max=GT_MAX),
        )
        self.batch_size_per_device = batch_size

    # ------------------------------------------------------------------
    def _make_dataloader(
        self,
        product_dir: str | None,
        max_products: int,
        shuffle: bool,
    ) -> DataLoader:
        """Helper to create a DataLoader for a given split."""
        hp = self.hparams  # type: ignore[attr-defined]
        patch_az, patch_rg = hp.patch_size  # type: ignore[attr-defined]
        total_az = patch_az + 2 * hp.azimuth_buffer  # type: ignore[attr-defined]
        cap = max_products if max_products > 0 else None

        if product_dir is None:
            log.warning("MAYA4DirDataModule: product_dir is None — returning empty DataLoader.")
            product_dir = "."  # rglob finds nothing, dataset will be empty

        dataset = RCMCSARDirDataset(
            product_dir=product_dir,
            azimuth_buffer=hp.azimuth_buffer,  # type: ignore[attr-defined]
            max_products=cap,
            data_dir=product_dir,
            transform=self.transforms,
            patch_size=(total_az, patch_rg),
            buffer=(0, 0),
            stride=(patch_az, patch_rg),
            level_from="rcmc",
            level_to="az",
            patch_mode="rectangular",
            complex_valued=False,
            positional_encoding=False,
            online=hp.online,  # type: ignore[attr-defined]
            samples_per_prod=hp.samples_per_prod,  # type: ignore[attr-defined]
            save_samples=True,
            verbose=False,
            use_balanced_sampling=False,
        )

        sampler = KPatchSampler(
            dataset,
            samples_per_prod=hp.samples_per_prod,  # type: ignore[attr-defined]
            shuffle_files=shuffle,
            patch_order="chunk",
            verbose=False,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            sampler=sampler,
            num_workers=hp.num_workers,  # type: ignore[attr-defined]
            pin_memory=hp.pin_memory,  # type: ignore[attr-defined]
            collate_fn=_collate_sar_batch,
        )

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        """Called by Lightning at the beginning of fit/test/predict to set up dataloaders."""
        hp = self.hparams  # type: ignore[attr-defined]
        if self.trainer is not None:
            world = self.trainer.world_size
            if hp.batch_size % world != 0:  # type: ignore[attr-defined]
                raise RuntimeError(
                    f"batch_size ({hp.batch_size}) must be divisible by world_size ({world})."  # type: ignore[attr-defined]
                )
            self.batch_size_per_device = hp.batch_size // world  # type: ignore[attr-defined]

        if stage in ("fit", None):
            self._train_loader = self._make_dataloader(
                hp.train_dir, hp.max_products_train, shuffle=True  # type: ignore[attr-defined]
            )
            self._val_loader = self._make_dataloader(
                hp.val_dir, hp.max_products_val, shuffle=False  # type: ignore[attr-defined]
            )
        if stage in ("test", None):
            self._test_loader = self._make_dataloader(
                hp.test_dir, hp.max_products_test, shuffle=False  # type: ignore[attr-defined]
            )

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return self._train_loader  # type: ignore[attr-defined]

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return self._val_loader  # type: ignore[attr-defined]

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        return self._test_loader  # type: ignore[attr-defined]
