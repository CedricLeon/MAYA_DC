from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


def to_complex_tensor(x: Tensor) -> Tensor:
    """Convert a SAR tensor to native complex dtype.

    Accepted layouts:
    - native complex tensor with shape ``(..., azimuth, range)``
    - real tensor with shape ``(..., 2, azimuth, range)`` where channel dim 1 is ``[real, imag]``
    """

    if torch.is_complex(x):
        return x

    if x.ndim == 3 and x.shape[0] == 2:
        return torch.complex(x[0, ...], x[1, ...])

    if x.ndim < 3:
        raise ValueError(f"Expected at least 3 dims for SAR tensor, got shape {tuple(x.shape)}")

    if x.shape[1] != 2:
        raise ValueError(
            "Real-valued SAR tensors must use shape (batch, 2, azimuth, range) with channel 1"
        )

    return torch.complex(x[:, 0, ...], x[:, 1, ...])


def _as_real_imag_channels(x: Tensor) -> Tensor:
    """Convert a complex tensor to real/imag channels."""
    if not torch.is_complex(x):
        raise TypeError("Expected a complex tensor")
    channel_dim = 1 if x.ndim >= 3 else 0
    return torch.stack((x.real, x.imag), dim=channel_dim)


def _coerce_crop_bounds(bounds: Any) -> tuple[int, int]:
    if isinstance(bounds, Tensor):
        if bounds.ndim == 1 and bounds.numel() == 2:
            return int(bounds[0].item()), int(bounds[1].item())
        if bounds.ndim == 2 and bounds.shape == (1, 2):
            return int(bounds[0, 0].item()), int(bounds[0, 1].item())
        raise ValueError(f"Unsupported crop tensor shape: {tuple(bounds.shape)}")

    if not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise TypeError(f"Crop bounds must be a 2-item sequence, got {type(bounds)!r}")

    values: list[int] = []
    for value in bounds:
        if isinstance(value, Tensor):
            values.append(int(value.item()))
        else:
            values.append(int(value))
    return values[0], values[1]


@dataclass
class AzimuthFocusMetadata:
    """Torch-native metadata required for azimuth compression."""

    slant_range_vec: Tensor
    azimuth_freq_hz: Tensor
    effective_velocity: Tensor
    wavelength_m: float
    crop_azimuth: tuple[int, int] = (0, 0)
    crop_range: tuple[int, int] = (0, 0)

    @classmethod
    def from_any(cls, metadata: "AzimuthFocusMetadata | Mapping[str, Any]") -> "AzimuthFocusMetadata":
        if isinstance(metadata, cls):
            return metadata

        if not isinstance(metadata, Mapping):
            raise TypeError(f"Unsupported focus metadata type: {type(metadata)!r}")

        return cls(
            slant_range_vec=metadata["slant_range_vec"],
            azimuth_freq_hz=metadata["azimuth_freq_hz"],
            effective_velocity=metadata["effective_velocity"],
            wavelength_m=float(metadata["wavelength_m"]),
            crop_azimuth=_coerce_crop_bounds(metadata.get("crop_azimuth", (0, 0))),
            crop_range=_coerce_crop_bounds(metadata.get("crop_range", (0, 0))),
        )

    def to_device(self, device: torch.device, dtype: torch.dtype) -> "AzimuthFocusMetadata":
        return AzimuthFocusMetadata(
            slant_range_vec=self.slant_range_vec.to(device=device, dtype=dtype),
            azimuth_freq_hz=self.azimuth_freq_hz.to(device=device, dtype=dtype),
            effective_velocity=self.effective_velocity.to(device=device, dtype=dtype),
            wavelength_m=self.wavelength_m,
            crop_azimuth=self.crop_azimuth,
            crop_range=self.crop_range,
        )


class TorchAzimuthFocus(nn.Module):
    """Differentiable azimuth compression path matching the Sarpyx math used in scripts."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def _build_filter(self, metadata: AzimuthFocusMetadata, like: Tensor) -> Tensor:
        md = metadata.to_device(like.device, torch.float64)

        if like.shape[-2] != md.azimuth_freq_hz.numel():
            raise ValueError(
                f"Azimuth dimension {like.shape[-2]} does not match frequency axis "
                f"{md.azimuth_freq_hz.numel()}"
            )
        if like.shape[-1] != md.slant_range_vec.numel():
            raise ValueError(
                f"Range dimension {like.shape[-1]} does not match slant-range vector "
                f"{md.slant_range_vec.numel()}"
            )
        if md.effective_velocity.numel() not in (1, like.shape[-2]):
            raise ValueError(
                "Effective velocity must be scalar or one value per azimuth line, got "
                f"{md.effective_velocity.numel()} values"
            )

        azimuth_freq = md.azimuth_freq_hz.reshape(-1)
        effective_velocity = md.effective_velocity.reshape(-1)
        if effective_velocity.numel() == 1:
            effective_velocity = effective_velocity.expand_as(azimuth_freq)

        d_inner = 1.0 - (
            (md.wavelength_m**2) * azimuth_freq.square()
        ) / (4.0 * effective_velocity.square().clamp_min(self.eps))
        d = torch.sqrt(d_inner.clamp_min(self.eps))

        phase = (
            4.0
            * torch.pi
            * d[:, None]
            * md.slant_range_vec[None, :]
            / md.wavelength_m
        )

        return torch.exp(1j * phase).to(device=like.device)

    @staticmethod
    def _crop(x: Tensor, crop_azimuth: tuple[int, int], crop_range: tuple[int, int]) -> Tensor:
        az_start, az_stop = crop_azimuth
        rg_start, rg_stop = crop_range

        az_slice = slice(az_start or None, None if az_stop == 0 else -az_stop)
        rg_slice = slice(rg_start or None, None if rg_stop == 0 else -rg_stop)
        return x[..., az_slice, rg_slice]

    def forward(
        self,
        rcmc_reconstruction: Tensor,
        metadata: AzimuthFocusMetadata | Mapping[str, Any],
    ) -> Tensor:
        complex_rcmc = to_complex_tensor(rcmc_reconstruction)
        md = AzimuthFocusMetadata.from_any(metadata)
        azimuth_filter = self._build_filter(md, complex_rcmc)
        filter_shape = (1,) * (complex_rcmc.ndim - 2) + azimuth_filter.shape
        azimuth_filter = azimuth_filter.view(filter_shape)

        doppler = torch.fft.fft(complex_rcmc, dim=-2)
        focused = torch.fft.ifft(doppler * azimuth_filter, dim=-2)
        return self._crop(focused, md.crop_azimuth, md.crop_range)


class IdentityFocus(nn.Module):
    """Fallback focus stage used when batches already contain the target domain."""

    def forward(
        self,
        rcmc_reconstruction: Tensor,
        metadata: AzimuthFocusMetadata | Mapping[str, Any] | None = None,
    ) -> Tensor:
        complex_rcmc = to_complex_tensor(rcmc_reconstruction)
        if metadata is None:
            return complex_rcmc

        md = AzimuthFocusMetadata.from_any(metadata)
        return TorchAzimuthFocus._crop(complex_rcmc, md.crop_azimuth, md.crop_range)


def maybe_get_complex_channels(x: Tensor) -> Tensor:
    """Return a tensor in the original representation after complex-domain ops."""
    if torch.is_complex(x):
        return _as_real_imag_channels(x)
    return x
