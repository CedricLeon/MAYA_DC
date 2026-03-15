"""MAYA4 SAR DataModule — simplified implementation.

Data pipeline
-------------
For each patch request (zfile, y, x):

    SARZarrDataset.__getitem__          reads rcmc and slc patches from zarr
         ↓  (overridden by RCMCSARDataset)
    RCMCSARDataset.__getitem__          also reads + slices + SWST-corrects metadata
                                        permutes tensors (H, W, 2) → (2, H, W)
                                        returns 5-tuple per sample
         ↓  (DataLoader calls collate)
    _collate_sar_batch                  stacks tensors into (B, 2, H, W)
                                        collects metadata / ephemeris as lists
         ↓
    (rcmc, slc, metadata_list, ephemeris_list, coords_list)

No wrapper class, no post-batch processing step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

import lightning
import rootutils
import torch
from torch import Tensor
from torch.utils.data import DataLoader

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from maya4 import (
    GT_MAX,
    GT_MIN,
    RC_MAX,
    RC_MIN,
    KPatchSampler,
    SampleFilter,
    SARTransform,
    SARZarrDataset,
)
from maya4.normalization import NormalizationModule

from src.utils.processing_utils import correct_swst_range_offset

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RCMCSARDataset(SARZarrDataset):
    """SARZarrDataset subclass for the RCMC compression pipeline.

    On top of what the parent does, each ``__getitem__`` call:

    1. Reads the metadata for the zarr file (cached per file).
    2. Slices metadata rows to the patch's azimuth range.
    3. Applies the SWST range offset when the patch doesn't start at column 0.
    4. Permutes tensors from ``(H, W, 2)`` to ``(2, H, W)``.
    5. Returns a 5-tuple: ``(rcmc, slc, metadata, ephemeris, coords)``.
    """

    def __init__(self, azimuth_buffer: int, max_products: int | None = None, **kwargs):
        # Pass max_products=None to the parent so it doesn't cap the file list early.
        # We apply the cap ourselves after _drop_files_without_ephemeris(), so that
        # files with broken ephemeris don't silently consume slots in the cap.
        super().__init__(max_products=999_999, **kwargs)
        self._azimuth_buffer = azimuth_buffer
        self._max_products_cap = max_products
        # Metadata cache: zfile path → (full_metadata_df, ephemeris_df)
        # Each DataLoader worker has its own copy — that is fine because
        # metadata DataFrames are tiny (a few hundred rows per file).
        self._meta_cache: dict[str, tuple[Any, Any]] = {}
        # Drop files whose metadata lacks ephemeris — CoarseRDA cannot focus
        # without satellite state vectors. Some zarr v2 products are affected.
        self._drop_files_without_ephemeris()

    def _drop_files_without_ephemeris(self) -> None:
        """Filter out zarr files that lack ephemeris data in their metadata."""
        all_files = self.get_files()
        valid = []
        for zfile in all_files:
            try:
                _, eph = self.get_metadata(str(zfile))
                if eph is not None and len(eph) > 0:
                    valid.append(zfile)
                else:
                    log.warning(
                        "RCMCSARDataset: dropping '%s' — ephemeris missing or empty", zfile
                    )
            except Exception as exc:
                log.warning("RCMCSARDataset: dropping '%s' — get_metadata raised %s", zfile, exc)
        n_dropped = len(all_files) - len(valid)
        # Apply max_products cap after filtering so bad files don't consume slots.
        if self._max_products_cap is not None:
            valid = valid[: self._max_products_cap]
        log.info(
            "RCMCSARDataset: %d/%d products valid after ephemeris check (dropped %d); "
            "using %d (cap=%s)",
            len(all_files) - n_dropped,
            len(all_files),
            n_dropped,
            len(valid),
            self._max_products_cap,
        )
        valid_set = {str(f) for f in valid}
        self._files = self._files[
            self._files["full_name"].apply(lambda p: str(p) in valid_set)
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    def _load_metadata(self, zfile: str) -> tuple[Any, Any]:
        """Load and cache full metadata for one zarr file."""
        if zfile not in self._meta_cache:
            meta, eph = self.get_metadata(zfile)
            self._meta_cache[zfile] = (meta, eph)
        return self._meta_cache[zfile]

    # ------------------------------------------------------------------
    def _slice_metadata(self, zfile: str, y: int, x: int, patch_height: int) -> tuple[Any, Any]:
        """Return metadata sliced to the patch's azimuth rows, with SWST corrected for range
        offset."""
        meta_full, eph = self._load_metadata(zfile)

        # Azimuth slice: rows y … y+patch_height (clamped to image height)
        az_end = min(y + patch_height, len(meta_full))
        meta = meta_full.iloc[y:az_end].reset_index(drop=True).copy()

        # Range (SWST) offset: if the patch doesn't start at range column 0,
        # shift the Sampling Window Start Time so CoarseRDA uses the right
        # near-range distance for its matched filter.
        correct_swst_range_offset(meta, x)

        return meta, eph

    # ------------------------------------------------------------------
    def __getitem__(self, idx: tuple[str, int, int]):
        """Return one sample: (rcmc, slc, metadata, ephemeris, coords)."""
        zfile, y, x = idx

        # Parent reads the patch and applies the SARTransform normalization.
        # Returns tensors of shape (H, W, 2) — last dim is [real, imag].
        patch_rcmc, patch_slc = super().__getitem__(idx)

        # Permute to (2, H, W) — the format the rest of the pipeline expects.
        rcmc: Tensor = patch_rcmc[..., :2].permute(2, 0, 1).float()
        slc: Tensor = patch_slc[..., :2].permute(2, 0, 1).float()

        # patch_height = full patch including azimuth buffers on both sides
        patch_height = rcmc.shape[1]  # H dimension
        meta, eph = self._slice_metadata(str(zfile), int(y), int(x), patch_height)

        coords = {"zfile": str(zfile), "y": int(y), "x": int(x)}
        return rcmc, slc, meta, eph, coords


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _collate_sar_batch(
    batch: list[tuple],
) -> tuple[Tensor, Tensor, list, list, list]:
    """Assemble N samples into one batch.

    Each sample is ``(rcmc, slc, metadata, ephemeris, coords)``.
    Tensors are stacked; metadata / ephemeris / coords stay as plain lists
    because PyTorch cannot stack DataFrames or dicts automatically.
    """
    rcmc_list, slc_list, meta_list, eph_list, coords_list = zip(*batch)
    return (
        torch.stack(list(rcmc_list)),  # (B, 2, H, W)
        torch.stack(list(slc_list)),  # (B, 2, H, W)
        list(meta_list),  # [DataFrame, ...]
        list(eph_list),  # [DataFrame, ...]
        list(coords_list),  # [dict, ...]
    )


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class MAYA4DataModule(lightning.LightningDataModule):
    """`LightningDataModule` for MAYA4 SAR data.

    Each batch is a 5-tuple::

        (rcmc, slc, metadata_list, ephemeris_list, coords_list)

    * ``rcmc``           – ``(B, 2, Az+2*buf, Rg)`` float32, normalised to [0, 1]
    * ``slc``            – ``(B, 2, Az+2*buf, Rg)`` float32, normalised to [0, 1]
    * ``metadata_list``  – one DataFrame per item, azimuth-sliced + SWST-corrected
    * ``ephemeris_list`` – one DataFrame per item
    * ``coords_list``    – one dict per item: ``{"zfile", "y", "x"}``
    """

    def __init__(
        self,
        data_dir: str = "data/",
        patch_size: tuple = (512, 512),
        azimuth_buffer: int = 512,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_parts: list[str] | None = None,
        val_parts: list[str] | None = None,
        test_parts: list[str] | None = None,
        years: list[int] | None = None,
        polarizations: list[str] | None = None,
        stripmap_modes: list[int] | None = None,
        max_products_train: int = 10,
        max_products_val: int = 5,
        max_products_test: int = 5,
        samples_per_prod: int = 100,
        online: bool = False,
    ) -> None:
        """
        Args:
            data_dir: Folder that holds the zarr product folders.
            patch_size: Core patch ``(azimuth_lines, range_samples)``.
                The dataset requests ``patch_size[0] + 2 * azimuth_buffer`` lines.
            azimuth_buffer: Extra lines on each side of the patch for CoarseRDA context.
                Constraint: ``patch_size[0] + 2 * azimuth_buffer`` must divide by 16.
            batch_size: Samples per batch (split across devices in DDP).
            num_workers: Worker processes for the DataLoader.
            pin_memory: Pin host memory for faster GPU transfer.
            train_parts / val_parts / test_parts: Geographic partitions (e.g. ``["PT1"]``).
            years: Filter products by acquisition year.
            polarizations: Filter by polarisation (e.g. ``["hh"]``).
            stripmap_modes: Filter by stripmap beam mode number.
            max_products_train/val/test: Max zarr files per split.
            samples_per_prod: Patches per file per epoch. ``0`` = all patches.
            online: Download missing zarr chunks from HuggingFace if ``True``.
        """
        super().__init__()

        # Validate the divisibility constraint required by the model's
        # four stride-2 convolution layers (16× total downsampling).
        total_az = patch_size[0] + 2 * azimuth_buffer
        assert total_az % 16 == 0, (
            f"patch_size[0] ({patch_size[0]}) + 2 × azimuth_buffer ({2 * azimuth_buffer}) "
            f"= {total_az}, which is not divisible by 16.\n"
            f"Valid examples: patch=512 buffer=128 → 768, patch=1024 buffer=256 → 1536."
        )

        # Safe defaults for mutable arguments
        train_parts = train_parts or ["PT1"]
        val_parts = val_parts or ["PT2"]
        test_parts = test_parts or ["PT4"]
        years = years or [2023]
        polarizations = polarizations or ["hh"]
        stripmap_modes = stripmap_modes or [1, 2, 3]

        self.save_hyperparameters(logger=False)

        # One shared SARTransform: normalises RCMC patches to [0,1] and
        # SLC patches to [0,1] using their respective physical value ranges.
        self.transforms = SARTransform(
            transform_rcmc=NormalizationModule(data_min=RC_MIN, data_max=RC_MAX),
            transform_az=NormalizationModule(data_min=GT_MIN, data_max=GT_MAX),
        )

        self.batch_size_per_device = batch_size

    # ------------------------------------------------------------------
    def _make_dataloader(
        self,
        parts: list[str],
        max_products: int,
        shuffle: bool,
    ) -> DataLoader:
        """Build one DataLoader for a given split."""
        hp = self.hparams  # type: ignore[attr-defined]

        patch_az = hp.patch_size[0]  # type: ignore[attr-defined]
        patch_rg = hp.patch_size[1]  # type: ignore[attr-defined]
        total_az = patch_az + 2 * hp.azimuth_buffer  # type: ignore[attr-defined]

        filters = SampleFilter(
            parts=parts,
            years=hp.years,  # type: ignore[attr-defined]
            polarizations=hp.polarizations,  # type: ignore[attr-defined]
            stripmap_modes=hp.stripmap_modes,  # type: ignore[attr-defined]
            # No zarr_versions filter here: _drop_files_without_ephemeris() handles
            # all culling in one place and logs how many files were dropped and why.
        )

        dataset = RCMCSARDataset(
            azimuth_buffer=hp.azimuth_buffer,  # type: ignore[attr-defined]
            # SARZarrDataset keyword arguments:
            data_dir=hp.data_dir,  # type: ignore[attr-defined]
            filters=filters,
            transform=self.transforms,
            patch_size=(total_az, patch_rg),
            buffer=(0, 0),  # no edge margin; azimuth_buffer already provides it
            stride=(patch_az, patch_rg),  # non-overlapping core patches
            level_from="rcmc",
            level_to="az",
            patch_mode="rectangular",
            complex_valued=False,  # split into real + imag channels
            positional_encoding=False,
            online=hp.online,  # type: ignore[attr-defined]
            max_products=max_products,
            samples_per_prod=hp.samples_per_prod,  # type: ignore[attr-defined]
            save_samples=True,
            verbose=False,
            use_balanced_sampling=False,  # we control diversity via train/val/test parts
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
        """Create the DataLoaders for each stage."""
        hp = self.hparams  # type: ignore[attr-defined]

        if self.trainer is not None:
            world = self.trainer.world_size
            if hp.batch_size % world != 0:  # type: ignore[attr-defined]
                raise RuntimeError(
                    f"batch_size ({hp.batch_size}) must be divisible "  # type: ignore[attr-defined]
                    f"by the number of GPUs ({world})."
                )
            self.batch_size_per_device = hp.batch_size // world  # type: ignore[attr-defined]

        if stage in ("fit", None):
            self._train_loader = self._make_dataloader(
                hp.train_parts, hp.max_products_train, shuffle=True  # type: ignore[attr-defined]
            )
            self._val_loader = self._make_dataloader(
                hp.val_parts, hp.max_products_val, shuffle=False  # type: ignore[attr-defined]
            )

        if stage in ("test", None):
            self._test_loader = self._make_dataloader(
                hp.test_parts, hp.max_products_test, shuffle=False  # type: ignore[attr-defined]
            )

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training set."""
        return self._train_loader  # type: ignore[attr-defined]

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation set."""
        return self._val_loader  # type: ignore[attr-defined]

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test set."""
        return self._test_loader  # type: ignore[attr-defined]


if __name__ == "__main__":
    dm = MAYA4DataModule()
    print(dm)
