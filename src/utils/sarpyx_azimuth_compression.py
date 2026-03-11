"""Full azimuth compression using sarpyx CoarseRDA pipeline.

This module implements the complete SAR azimuth focusing pipeline matching the behavior in
scripts/sarpyx_azimuth_compression.py.
"""

import numpy as np
import torch
from sarpyx.processor.algorithms.constants import TX_WAVELENGTH_M
from sarpyx.processor.core.focus import CoarseRDA
from torch import Tensor


def full_azimuth_compress_batch(
    rcmc_batch: Tensor,
    metadata_batch: dict,
    ephemeris: dict,
    buffer_size: int = 500,
    device: str = "cpu",
) -> Tensor:
    """Perform full SAR azimuth compression using sarpyx CoarseRDA.

    This implements the complete azimuth focusing pipeline that matches
    the behavior in scripts/sarpyx_azimuth_compression.py.

    Args:
        rcmc_batch: RCMC data with shape (B, C=2, Az, Rg) where Az includes buffers
                   Channels are [real, imag]
        metadata_batch: Dictionary with SAR metadata per batch item
        ephemeris: Ephemeris data (satellite positions/velocities)
        buffer_size: Size of buffer zone in azimuth direction
        device: Device for tensor operations

    Returns:
        SLC data with shape (B, C=2, Az-2*buffer, Rg) (buffers removed)
    """
    B, C, Az, Rg = rcmc_batch.shape
    Az_core = Az - 2 * buffer_size

    # Convert to numpy and combine real/imag to complex
    rcmc_np = rcmc_batch.cpu().numpy()
    if C == 2:
        # Combine real and imaginary channels
        rcmc_complex = rcmc_np[:, 0, :, :] + 1j * rcmc_np[:, 1, :, :]
    else:
        rcmc_complex = rcmc_np.squeeze(1)  # Assume already complex

    # Initialize output
    slc_batch_complex = np.zeros((B, Az_core, Rg), dtype=np.complex128)

    # Process each sample in batch
    for b in range(B):
        rcmc_patch = rcmc_complex[b]  # Shape: (Az, Rg)

        # Get metadata and ephemeris for this sample
        metadata = metadata_batch[b] if isinstance(metadata_batch, list) else metadata_batch
        eph = ephemeris[b] if isinstance(ephemeris, list) else ephemeris

        # Create mock raw_data for CoarseRDA
        mock_raw_data = {
            "echo": rcmc_patch,
            "metadata": metadata,
            "ephemeris": eph,
        }

        # Initialize processor
        processor = CoarseRDA(mock_raw_data, verbose=False, memory_efficient=False)

        # Step 1: Transform to Doppler domain (Azimuth FFT)
        processor.radar_data = np.fft.fft(processor.radar_data, axis=0)

        # Step 2: Compute effective velocities
        processor._compute_effective_velocities()

        # Step 2b: Compute D parameter
        processor.wavelength = TX_WAVELENGTH_M

        # Generate azimuth frequency values
        processor.az_freq_vals = np.arange(
            -processor.az_sample_freq / 2,
            processor.az_sample_freq / 2,
            processor.az_sample_freq / processor.len_az_line,
        )
        if len(processor.az_freq_vals) != processor.len_az_line:
            processor.az_freq_vals = np.linspace(
                -processor.az_sample_freq / 2,
                processor.az_sample_freq / 2,
                processor.len_az_line,
                endpoint=False,
            )

        # D = sqrt(1 - (lambda*f)^2 / (4*V^2))
        mean_effective_velocities = np.mean(processor.effective_velocities, axis=1)
        processor.D = np.sqrt(
            1
            - (processor.wavelength**2 * processor.az_freq_vals**2)
            / (4 * mean_effective_velocities**2)
        )

        # Step 3: Perform azimuth compression (filter * IFFT)
        processor._perform_azimuth_compression_efficient()

        # Get result (already in time domain)
        slc_full = processor.radar_data

        # Step 4: Remove buffers
        slc_core = slc_full[buffer_size : buffer_size + Az_core, :]
        slc_batch_complex[b] = slc_core

    # Convert back to torch tensor with real/imag channels
    slc_real = torch.from_numpy(slc_batch_complex.real).to(device)
    slc_imag = torch.from_numpy(slc_batch_complex.imag).to(device)
    slc_tensor = torch.stack([slc_real, slc_imag], dim=1)  # (B, C=2, Az_core, Rg)

    return slc_tensor


class SarpyxAzimuthCompressor:
    """Wrapper class for azimuth compression using sarpyx.

    This provides a cleaner interface for the neural network pipeline.
    """

    def __init__(self, buffer_size: int = 500):
        """Initialize azimuth compressor.

        Args:
            buffer_size: Size of buffer zone in azimuth direction
        """
        self.buffer_size = buffer_size

    def __call__(self, rcmc_batch: Tensor, metadata_batch: dict, ephemeris: dict) -> Tensor:
        """Compress a batch of RCMC data.

        Args:
            rcmc_batch: RCMC data (B, C=2, Az, Rg)
            metadata_batch: Metadata dictionary
            ephemeris: Ephemeris data

        Returns:
            SLC data (B, C=2, Az-2*buffer, Rg)
        """
        return full_azimuth_compress_batch(
            rcmc_batch,
            metadata_batch,
            ephemeris,
            buffer_size=self.buffer_size,
            device=rcmc_batch.device,
        )
