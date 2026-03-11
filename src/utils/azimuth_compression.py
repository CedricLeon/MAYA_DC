"""Azimuth compression utilities for SAR processing."""

import numpy as np
import torch
from torch import Tensor


def azimuth_compress_batch(
    rcmc_batch: Tensor,
    metadata: dict,
    buffer_size: int = 500,
) -> Tensor:
    """Perform azimuth compression on a batch of RCMC patches.

    This implements the azimuth focusing step that converts RCMC data to SLC.
    The input includes buffer zones in azimuth that are removed after processing.

    Args:
        rcmc_batch: RCMC data with shape (B, C, Az, Rg) where Az includes buffers
        metadata: Dictionary with SAR parameters (velocity, wavelength, etc.)
        buffer_size: Size of buffer zone in azimuth direction

    Returns:
        SLC data with shape (B, C, Az-2*buffer, Rg) (buffers removed)
    """
    B, C, Az, Rg = rcmc_batch.shape
    device = rcmc_batch.device

    # Extract core region size (without buffers)
    Az_core = Az - 2 * buffer_size

    # Convert to numpy for processing (PyTorch FFT on complex is not fully supported)
    rcmc_np = rcmc_batch.cpu().numpy()

    # Initialize output
    slc_batch = np.zeros((B, C, Az_core, Rg), dtype=rcmc_np.dtype)

    # Process each sample in batch
    for b in range(B):
        for c in range(C):
            rcmc_patch = rcmc_np[b, c]  # Shape: (Az, Rg)

            # 1. Transform to Range-Doppler domain (Azimuth FFT)
            rcmc_doppler = np.fft.fft(rcmc_patch, axis=0)

            # 2. Apply azimuth compression (this is simplified)
            # In full implementation, this would use matched filtering with
            # azimuth reference function based on metadata

            # For now: azimuth IFFT to get SLC
            slc_full = np.fft.ifft(rcmc_doppler, axis=0)

            # 3. Remove buffer zones
            slc_core = slc_full[buffer_size : buffer_size + Az_core, :]

            slc_batch[b, c] = slc_core

    # Convert back to torch tensor
    slc_tensor = torch.from_numpy(slc_batch).to(device)

    return slc_tensor


def azimuth_compress_with_metadata(
    rcmc_batch: Tensor,
    buffer_size: int = 500,
) -> Tensor:
    """Interim azimuth compression that preserves complex phase information.

    Performs azimuth FFT → IFFT on the *complex* SAR signal (treating the two
    real/imag channels as a single complex channel).  This is mathematically an
    identity transform, but it correctly handles complex phase — unlike
    ``simple_azimuth_compress`` which called ``.abs()`` and destroyed all phase
    information.

    The network trained with this function learns to reconstruct RCMC data such
    that its identity-transformed version resembles the ground-truth SLC.  The
    gradients are physically meaningful only after the matched azimuth filter
    (from CoarseRDA / sarpyx) is wired in.

    # TODO: replace with full_azimuth_compress_batch once metadata is plumbed
    #       through the data pipeline (see BUG 2 in the implementation notes).

    Args:
        rcmc_batch: RCMC data (B, C=2, Az, Rg) – channels are [real, imag]
        buffer_size: Azimuth buffer size to remove from both ends

    Returns:
        Azimuth-compressed data (B, 2, Az-2*buffer, Rg) – channels are [real, imag]
    """
    B, C, Az, Rg = rcmc_batch.shape
    Az_core = Az - 2 * buffer_size
    assert C == 2, f"Expected 2 channels (real, imag), got {C}"

    # Combine real/imag into a single complex tensor: (B, Az, Rg)
    x_complex = torch.complex(rcmc_batch[:, 0, :, :], rcmc_batch[:, 1, :, :])

    # Azimuth FFT → IFFT (identity transform)
    # TODO: Apply the matched azimuth filter in Doppler domain here using metadata.
    x_doppler = torch.fft.fft(x_complex, dim=1)
    x_focused = torch.fft.ifft(x_doppler, dim=1)

    # Remove azimuth buffer
    x_core = x_focused[:, buffer_size : buffer_size + Az_core, :]  # (B, Az_core, Rg)

    # Return as real/imag channels (B, 2, Az_core, Rg)
    dtype = rcmc_batch.dtype
    return torch.stack([x_core.real.to(dtype), x_core.imag.to(dtype)], dim=1)


def simple_azimuth_compress(rcmc_batch: Tensor, buffer_size: int = 500) -> Tensor:
    """**Deprecated – do not use in new code.**

    This function applies FFT → IFFT (an identity) on separate real/imag channels
    and then calls ``.abs()``, which destroys all phase information.  It exists
    only for backward-compatibility reference.

    Use :func:`azimuth_compress_with_metadata` instead.

    Args:
        rcmc_batch: RCMC data (B, C, Az, Rg)
        buffer_size: Buffer size in azimuth

    Returns:
        Magnitude-only SLC data (B, C, Az-2*buffer, Rg)  ← phase information lost!
    """
    import warnings

    warnings.warn(
        "simple_azimuth_compress applies .abs() and destroys phase. "
        "Use azimuth_compress_with_metadata instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    B, C, Az, Rg = rcmc_batch.shape
    Az_core = Az - 2 * buffer_size

    rcmc_doppler = torch.fft.fft(rcmc_batch, dim=2)
    slc_full = torch.fft.ifft(rcmc_doppler, dim=2).abs()  # phase destroyed

    slc_core = slc_full[:, :, buffer_size : buffer_size + Az_core, :]
    return slc_core
