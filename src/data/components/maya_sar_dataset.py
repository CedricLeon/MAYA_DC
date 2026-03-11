from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.sar_environment import ensure_sar_environment


def _import_sarpyx_runtime():
    from sarpyx.processor.algorithms.constants import RANGE_DECIMATION_MAP, TX_WAVELENGTH_M
    from sarpyx.processor.core.focus import CoarseRDA
    from sarpyx.utils.zarr_utils import ProductHandler

    return ProductHandler, CoarseRDA, RANGE_DECIMATION_MAP, TX_WAVELENGTH_M


def _processing_helpers():
    from src.utils.processing_utils import load_partitioned_dataframe, scan_available_data_extent

    return load_partitioned_dataframe, scan_available_data_extent


def _open_product_handler(product_path: Path):
    product_handler_cls, _, _, _ = _import_sarpyx_runtime()
    return product_handler_cls(product_path)


def _to_complex_channels(array: np.ndarray) -> Tensor:
    if not np.iscomplexobj(array):
        raise ValueError(f"Expected complex-valued SAR array, got dtype {array.dtype}")

    channels = np.stack((array.real, array.imag), axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(channels)


def _bounds_to_slice(bounds: tuple[int, int]) -> slice:
    return slice(bounds[0], bounds[1])


def _window_starts(start: int, stop: int, size: int, stride: int) -> list[int]:
    last_start = stop - size
    if last_start < start:
        return []

    starts = list(range(start, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@dataclass(frozen=True)
class SARPatchSample:
    product_path: str
    full_azimuth_size: int
    input_azimuth: tuple[int, int]
    input_range: tuple[int, int]
    target_azimuth: tuple[int, int]
    target_range: tuple[int, int]
    crop_azimuth: tuple[int, int]
    crop_range: tuple[int, int]


@lru_cache(maxsize=64)
def _load_product_layout(
    product_path_str: str,
    level_from: str,
    level_to: str,
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    _, scan_available_data_extent = _processing_helpers()
    product_path = Path(product_path_str)
    handler = _open_product_handler(product_path)

    source = handler.get_array(level_from)
    target = handler.get_array(level_to)
    source_az_slice, source_rg_slice = scan_available_data_extent(
        product_path, array_name=level_from, verbose=False
    )
    target_az_slice, target_rg_slice = scan_available_data_extent(
        product_path, array_name=level_to, verbose=False
    )

    effective_azimuth = (
        max(source_az_slice.start, target_az_slice.start),
        min(source_az_slice.stop, target_az_slice.stop, source.shape[0], target.shape[0]),
    )
    effective_range = (
        max(source_rg_slice.start, target_rg_slice.start),
        min(source_rg_slice.stop, target_rg_slice.stop, source.shape[1], target.shape[1]),
    )

    return int(source.shape[0]), effective_azimuth, effective_range


@lru_cache(maxsize=16)
def _load_product_tables(product_path_str: str):
    load_partitioned_dataframe, _ = _processing_helpers()
    handler = _open_product_handler(Path(product_path_str))
    return (
        load_partitioned_dataframe(handler, "metadata"),
        load_partitioned_dataframe(handler, "ephemeris"),
    )


def _build_focus_metadata(
    product_path: Path,
    input_azimuth: tuple[int, int],
    input_range: tuple[int, int],
    full_azimuth_size: int,
    crop_azimuth: tuple[int, int],
    crop_range: tuple[int, int],
) -> dict[str, Any]:
    _, CoarseRDA, RANGE_DECIMATION_MAP, TX_WAVELENGTH_M = _import_sarpyx_runtime()
    metadata_full, ephemeris_full = _load_product_tables(str(product_path))

    az_slice = _bounds_to_slice(input_azimuth)
    rg_slice = _bounds_to_slice(input_range)
    metadata_slice = metadata_full.copy()

    if az_slice.start > 0 or az_slice.stop < full_azimuth_size:
        metadata_slice = metadata_slice.iloc[az_slice].reset_index(drop=True)

    if rg_slice.start > 0:
        meta_row = metadata_slice.iloc[0]
        rgdec = meta_row.get("Range Decimation", meta_row.get("range_decimation"))
        if rgdec is None:
            raise ValueError("Range Decimation not found in metadata; cannot derive focus metadata")

        range_freq = float(RANGE_DECIMATION_MAP[int(rgdec)])
        if "swst" in metadata_slice.columns:
            swst_col = "swst"
        elif "SWST" in metadata_slice.columns:
            swst_col = "SWST"
        else:
            raise ValueError("SWST not found in metadata; cannot derive focus metadata")
        metadata_slice[swst_col] = metadata_slice[swst_col] + (rg_slice.start / range_freq)

    mock_raw_data = {
        "echo": np.zeros((az_slice.stop - az_slice.start, rg_slice.stop - rg_slice.start), dtype=np.complex64),
        "metadata": metadata_slice,
        "ephemeris": ephemeris_full,
    }
    processor = CoarseRDA(mock_raw_data, verbose=False, memory_efficient=False)
    processor._compute_effective_velocities()

    azimuth_freq_hz = np.arange(
        -processor.az_sample_freq / 2,
        processor.az_sample_freq / 2,
        processor.az_sample_freq / processor.len_az_line,
    )
    if len(azimuth_freq_hz) != processor.len_az_line:
        azimuth_freq_hz = np.linspace(
            -processor.az_sample_freq / 2,
            processor.az_sample_freq / 2,
            processor.len_az_line,
            endpoint=False,
        )

    effective_velocity = np.mean(processor.effective_velocities, axis=1)

    return {
        "slant_range_vec": torch.from_numpy(np.asarray(processor.slant_range_vec, dtype=np.float64)),
        "azimuth_freq_hz": torch.from_numpy(np.asarray(azimuth_freq_hz, dtype=np.float64)),
        "effective_velocity": torch.from_numpy(np.asarray(effective_velocity, dtype=np.float64)),
        "wavelength_m": float(TX_WAVELENGTH_M),
        "crop_azimuth": crop_azimuth,
        "crop_range": crop_range,
    }


def sar_batch_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError(
            "The current SAR training path only supports batch_size=1 because focus metadata is "
            "computed per sample."
        )

    sample = dict(batch[0])
    sample["rcmc_input"] = sample["rcmc_input"].unsqueeze(0)
    sample["slc_target"] = sample["slc_target"].unsqueeze(0)
    return sample


class MAYASARPatchDataset(Dataset[dict[str, Any]]):
    """Patch dataset for MAYA/Sarpyx products emitting differentiable SAR training batches."""

    def __init__(
        self,
        data_dir: str | Path,
        partitions: Sequence[str],
        patch_size: tuple[int, int] = (512, 512),
        stride: tuple[int, int] | None = None,
        azimuth_margin: int = 500,
        range_margin: int = 0,
        level_from: str = "rcmc",
        level_to: str = "az",
        max_products: int | None = None,
        enforce_full_context: bool = True,
    ) -> None:
        super().__init__()
        ensure_sar_environment(project_root=Path(__file__).resolve().parents[3])

        self.data_dir = Path(data_dir)
        self.partitions = tuple(partitions)
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.azimuth_margin = azimuth_margin
        self.range_margin = range_margin
        self.level_from = level_from
        self.level_to = level_to
        self.max_products = max_products
        self.enforce_full_context = enforce_full_context

        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(
                f"No SAR samples were indexed under {self.data_dir} for partitions {self.partitions}."
            )

    def _discover_product_paths(self) -> list[Path]:
        products: list[Path] = []
        for partition in self.partitions:
            partition_dir = self.data_dir / partition
            if not partition_dir.exists():
                raise FileNotFoundError(f"Partition directory not found: {partition_dir}")
            products.extend(sorted(path for path in partition_dir.glob("**/*.zarr") if path.is_dir()))

        if self.max_products is not None:
            products = products[: self.max_products]
        return products

    def _iter_product_samples(self, product_path: Path) -> Iterable[SARPatchSample]:
        full_azimuth_size, effective_azimuth, effective_range = _load_product_layout(
            str(product_path), self.level_from, self.level_to
        )

        target_azimuth_start = effective_azimuth[0] + (
            self.azimuth_margin if self.enforce_full_context else 0
        )
        target_azimuth_stop = effective_azimuth[1] - (
            self.azimuth_margin if self.enforce_full_context else 0
        )
        target_range_start = effective_range[0] + (self.range_margin if self.enforce_full_context else 0)
        target_range_stop = effective_range[1] - (self.range_margin if self.enforce_full_context else 0)

        azimuth_starts = _window_starts(
            target_azimuth_start,
            target_azimuth_stop,
            self.patch_size[0],
            self.stride[0],
        )
        range_starts = _window_starts(
            target_range_start,
            target_range_stop,
            self.patch_size[1],
            self.stride[1],
        )

        for az_start in azimuth_starts:
            for rg_start in range_starts:
                target_azimuth = (az_start, az_start + self.patch_size[0])
                target_range = (rg_start, rg_start + self.patch_size[1])
                input_azimuth = (
                    max(effective_azimuth[0], target_azimuth[0] - self.azimuth_margin),
                    min(effective_azimuth[1], target_azimuth[1] + self.azimuth_margin),
                )
                input_range = (
                    max(effective_range[0], target_range[0] - self.range_margin),
                    min(effective_range[1], target_range[1] + self.range_margin),
                )

                yield SARPatchSample(
                    product_path=str(product_path),
                    full_azimuth_size=full_azimuth_size,
                    input_azimuth=input_azimuth,
                    input_range=input_range,
                    target_azimuth=target_azimuth,
                    target_range=target_range,
                    crop_azimuth=(
                        target_azimuth[0] - input_azimuth[0],
                        input_azimuth[1] - target_azimuth[1],
                    ),
                    crop_range=(
                        target_range[0] - input_range[0],
                        input_range[1] - target_range[1],
                    ),
                )

    def _index_samples(self) -> list[SARPatchSample]:
        samples: list[SARPatchSample] = []
        for product_path in self._discover_product_paths():
            samples.extend(self._iter_product_samples(product_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        product_path = Path(sample.product_path)
        handler = _open_product_handler(product_path)

        input_azimuth_slice = _bounds_to_slice(sample.input_azimuth)
        input_range_slice = _bounds_to_slice(sample.input_range)
        target_azimuth_slice = _bounds_to_slice(sample.target_azimuth)
        target_range_slice = _bounds_to_slice(sample.target_range)

        rcmc_input = handler.get_array(self.level_from)[input_azimuth_slice, input_range_slice]
        slc_target = handler.get_array(self.level_to)[target_azimuth_slice, target_range_slice]
        focus_metadata = _build_focus_metadata(
            product_path=product_path,
            input_azimuth=sample.input_azimuth,
            input_range=sample.input_range,
            full_azimuth_size=sample.full_azimuth_size,
            crop_azimuth=sample.crop_azimuth,
            crop_range=sample.crop_range,
        )

        return {
            "rcmc_input": _to_complex_channels(np.asarray(rcmc_input)),
            "slc_target": _to_complex_channels(np.asarray(slc_target)),
            "focus_metadata": focus_metadata,
        }
