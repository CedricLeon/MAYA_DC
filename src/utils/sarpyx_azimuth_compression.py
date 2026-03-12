"""Standalone differentiable azimuth compression for SAR data.

Replaces the CoarseRDA-based loop in ``full_azimuth_compress_batch`` with:

1. A standalone numpy/scipy filter computation (``compute_azimuth_filter``)
   that mirrors the relevant CoarseRDA methods but does *not* initialise the
   full processor (no tx-replica generation, no range filter).
2. A fully differentiable ``torch.fft`` application so that gradients flow
   from the SLC loss back into the RCMC decoder (``g_s``).

Filter computation subfunctions mirror CoarseRDA methods in
``srp/sarpyx/processor/core/focus.py``:

  ``_initialize_timing_parameters``  ←→  CoarseRDA._initialize_timing_parameters
                                          + range_sample_freq from _generate_tx_replica
  ``_calculate_spacecraft_dynamics`` ←→  CoarseRDA._calculate_spacecraft_dynamics
  ``_compute_ground_velocities``     ←→  CoarseRDA._compute_ground_velocities
  ``compute_azimuth_filter``         ←→  CoarseRDA.get_azimuth_filter
                                          (+ _compute_effective_velocities)
"""

import math
from typing import Any

import numpy as np
import torch
from sarpyx.processor.core.code2physical import range_dec_to_sample_rate
from sarpyx.processor.core.constants import (
    F_REF,
    SPEED_OF_LIGHT_MPS,
    TX_WAVELENGTH_M,
    WGS84_SEMI_MAJOR_AXIS_M,
    WGS84_SEMI_MINOR_AXIS_M,
)
from scipy.interpolate import interp1d
from torch import Tensor

# ---------------------------------------------------------------------------
# Filter computation helpers
# One call per batch item; pure numpy/scipy — no gradient needed.
# ---------------------------------------------------------------------------


def _initialize_timing_parameters(
    metadata: Any,
    len_range_line: int,
) -> tuple:
    """Compute azimuth sample frequency and slant-range vector from metadata.

    Mirrors ``CoarseRDA._initialize_timing_parameters``.
    The ``range_sample_freq`` used here is extracted the same way as in
    ``CoarseRDA._generate_tx_replica`` (``range_dec_to_sample_rate``).

    Args:
        metadata:        Pandas DataFrame for the SAR patch.
        len_range_line:  Number of range samples in the patch.

    Returns:
        az_sample_freq:  Azimuth sampling frequency (Hz).
        slant_range_vec: Slant range per range sample (m), shape ``(len_range_line,)``.
    """
    # -- column search (mirrors CoarseRDA._initialize_timing_parameters) ----
    meta_lower = {col.lower(): col for col in metadata.columns}

    def _find_col(param, candidates):
        for name in candidates:
            if name in meta_lower:
                return meta_lower[name]
        raise KeyError(
            f"Cannot find metadata column for '{param}'. "
            f"Tried {candidates}. Available: {list(metadata.columns)}"
        )

    pri_col = _find_col("pri", ["pri", "pulse_repetition_interval"])
    rank_col = _find_col("rank", ["rank"])
    swst_col = _find_col("swst", ["swst", "sampling_window_start_time", "start_time"])

    pri = float(metadata[pri_col].iloc[0])
    rank = float(metadata[rank_col].iloc[0])
    range_start_time_base = float(metadata[swst_col].iloc[0])

    # range_sample_freq from range decimation code (CoarseRDA._generate_tx_replica)
    rgdec = int(metadata["range_decimation"].unique()[0])
    range_sample_freq = range_dec_to_sample_rate(rgdec)

    suppressed_data_time = 320.0 / (8.0 * F_REF)
    range_start_time = range_start_time_base + suppressed_data_time

    az_sample_freq = 1.0 / pri
    range_sample_period = 1.0 / range_sample_freq

    sample_num = np.arange(len_range_line, dtype=np.float64)
    fast_time_vec = range_start_time + range_sample_period * sample_num
    slant_range_vec = (rank * pri + fast_time_vec) * SPEED_OF_LIGHT_MPS / 2.0

    return az_sample_freq, slant_range_vec


def _calculate_spacecraft_dynamics(metadata: Any, ephemeris: Any) -> tuple:
    """Interpolate spacecraft velocities and positions at azimuth line times.

    Mirrors ``CoarseRDA._calculate_spacecraft_dynamics``.
    The ephemeris ``time_stamp`` is scaled by ``1/2^24`` before interpolation,
    matching ``CoarseRDA._load_data``.

    Args:
        metadata:   Pandas DataFrame (must have ``coarse_time``, ``fine_time``).
        ephemeris:  Pandas DataFrame (must have ``time_stamp``, ``vx``, ``vy``,
                    ``vz``, ``x``, ``y``, ``z``).

    Returns:
        space_velocities: ECEF speed per azimuth line, shape ``(az_lines,)``.
        positions:        ECEF position per azimuth line, shape ``(az_lines, 3)``.
    """
    eph = ephemeris.copy()
    eph["time_stamp"] = eph["time_stamp"] / 2**24  # CoarseRDA._load_data

    ecef_vels = eph.apply(lambda r: math.sqrt(r["vx"] ** 2 + r["vy"] ** 2 + r["vz"] ** 2), axis=1)

    ts = eph["time_stamp"].values
    v = ecef_vels.values
    x = eph["x"].values
    y = eph["y"].values
    z = eph["z"].values

    order = np.argsort(ts)
    ts, v, x, y, z = ts[order], v[order], x[order], y[order], z[order]

    metadata_times = metadata.apply(lambda r: r["coarse_time"] + r["fine_time"], axis=1).values

    kw = dict(kind="linear", bounds_error=False)
    space_velocities = np.asarray(interp1d(ts, v, fill_value=(v[0], v[-1]), **kw)(metadata_times))
    positions = np.column_stack(
        [
            interp1d(ts, x, fill_value=(x[0], x[-1]), **kw)(metadata_times),
            interp1d(ts, y, fill_value=(y[0], y[-1]), **kw)(metadata_times),
            interp1d(ts, z, fill_value=(z[0], z[-1]), **kw)(metadata_times),
        ]
    )

    return space_velocities, positions


def _compute_ground_velocities(
    space_velocities: np.ndarray,
    positions: np.ndarray,
    slant_range_vec: np.ndarray,
) -> np.ndarray:
    """Compute effective velocities using the WGS-84 Earth model.

    Mirrors ``CoarseRDA._compute_ground_velocities``.

    Args:
        space_velocities: ECEF speed per azimuth line, shape ``(az,)``.
        positions:        ECEF position per azimuth line, shape ``(az, 3)``.
        slant_range_vec:  Slant range per range sample, shape ``(rg,)``.

    Returns:
        effective_velocities: Shape ``(az, rg)``.
    """
    a = float(WGS84_SEMI_MAJOR_AXIS_M)
    b = float(WGS84_SEMI_MINOR_AXIS_M)

    H = np.linalg.norm(positions, axis=1)  # (az,) – orbital radius
    W = space_velocities / H  # (az,) – angular velocity

    xy_dist = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    lat = np.arctan2(positions[:, 2], xy_dist)

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    local_earth_rad = np.sqrt(
        (a**4 * cos_lat**2 + b**4 * sin_lat**2) / (a**2 * cos_lat**2 + b**2 * sin_lat**2)
    )

    # Broadcast (az, 1) × (1, rg)
    Re = local_earth_rad[:, np.newaxis]
    H_ = H[:, np.newaxis]
    W_ = W[:, np.newaxis]
    R = slant_range_vec[np.newaxis, :]

    cos_beta = np.clip((Re**2 + H_**2 - R**2) / (2.0 * Re * H_), -1.0, 1.0)
    ground_velocities = Re * W_ * cos_beta
    effective_velocities = np.sqrt(
        np.maximum(space_velocities[:, np.newaxis] * ground_velocities, 0.0)
    )

    return effective_velocities  # (az, rg)


def compute_azimuth_filter(
    metadata: Any,
    ephemeris: Any,
    len_az_line: int,
    len_range_line: int,
) -> np.ndarray:
    """Compute the azimuth compression filter H for a single SAR patch.

    Mirrors ``CoarseRDA.get_azimuth_filter`` (plus the velocity pipeline it
    depends on: ``_compute_effective_velocities`` → ``get_rcmc``).

    H is a **constant w.r.t. the radar data** — it depends only on orbit
    geometry and sensor timing.  Converting H to a torch tensor and applying
    it via ``torch.fft`` therefore preserves gradients through the data path.

    Args:
        metadata:        Pandas DataFrame for the patch.
        ephemeris:       Pandas DataFrame with satellite state vectors.
        len_az_line:     Number of azimuth lines (including azimuth buffer).
        len_range_line:  Number of range samples.

    Returns:
        H: Complex azimuth filter, shape ``(len_az_line, len_range_line)``,
           dtype ``complex128``.
    """
    az_sample_freq, slant_range_vec = _initialize_timing_parameters(metadata, len_range_line)
    space_velocities, positions = _calculate_spacecraft_dynamics(metadata, ephemeris)
    effective_velocities = _compute_ground_velocities(space_velocities, positions, slant_range_vec)

    az_freq_vals = np.linspace(
        -az_sample_freq / 2.0,
        az_sample_freq / 2.0,
        len_az_line,
        endpoint=False,
    )
    mean_V = np.mean(effective_velocities, axis=1)  # (az,)

    # Cosine of instantaneous squint angle (CoarseRDA.get_rcmc / get_azimuth_filter)
    D = np.sqrt(
        np.maximum(
            1.0 - (TX_WAVELENGTH_M**2 * az_freq_vals**2) / (4.0 * mean_V**2),
            0.0,
        )
    )

    # H[az, rg] = exp(4j·π·R[rg]·D[az] / λ)   (CoarseRDA.get_azimuth_filter)
    H = np.exp(
        4j
        * np.pi
        * slant_range_vec[np.newaxis, :]  # (1, rg)
        * D[:, np.newaxis]  # (az, 1)
        / TX_WAVELENGTH_M
    )  # (az, rg), complex128

    return H


# ---------------------------------------------------------------------------
# Differentiable batch azimuth compression
# ---------------------------------------------------------------------------


def full_azimuth_compress_batch(
    rcmc_batch: Tensor,
    metadata_batch: Any,
    ephemeris_batch: Any,
    buffer_size: int = 500,
    device: Any = "cpu",
) -> Tensor:
    """Azimuth-focus a batch of RCMC patches, preserving torch gradients.

    Pipeline per batch item:

    1. ``compute_azimuth_filter`` builds constant filter ``H`` from metadata
       and ephemeris (numpy/scipy — no CoarseRDA instantiation).
    2. The radar data path stays entirely in PyTorch:
       ``FFT_az(x) → ×H → IFFT_az`` — gradients flow back through ``x``.
    3. The azimuth buffer is stripped and real/imag channels are returned.

    Gradient note: H is a constant tensor (no ``grad_fn``). The product
    ``FFT(x) * H`` preserves the ``grad_fn`` of ``FFT(x)``, so
    ``d(SLC)/d(x) = IFFT(H)`` is correctly propagated during backprop.

    Args:
        rcmc_batch:      ``(B, 2, Az+2·buf, Rg)`` real/imag channels.
                         May have ``requires_grad=True``.
        metadata_batch:  List of per-item metadata DataFrames (length B).
        ephemeris_batch: List of per-item ephemeris DataFrames (length B).
        buffer_size:     Azimuth buffer samples to strip from each side.
        device:          Target device for the output tensor.

    Returns:
        SLC tensor ``(B, 2, Az, Rg)`` on ``device``, with gradients if
        ``rcmc_batch.requires_grad``.
    """
    B, C, Az, Rg = rcmc_batch.shape
    Az_core = Az - 2 * buffer_size
    assert C == 2, f"Expected 2 channels (real/imag), got {C}"

    # Build complex input via view_as_complex — a true view op with well-tested autograd.
    # (B, 2, Az, Rg) → (B, Az, Rg, 2) → view as (B, Az, Rg) complex.
    # Avoids torch.complex() on non-contiguous slices, which has less autograd coverage.
    x_ri = rcmc_batch.permute(0, 2, 3, 1).contiguous()  # (B, Az, Rg, 2)
    x_c = torch.view_as_complex(x_ri)  # (B, Az, Rg) complex

    # Azimuth FFT (differentiable)
    X = torch.fft.fft(x_c, dim=1)  # (B, Az, Rg)

    # Per-item filter application.
    # Collect in a list — avoids in-place writes on a graph tensor.
    slc_ffts = []
    for b in range(B):
        meta = metadata_batch[b] if isinstance(metadata_batch, list) else metadata_batch
        eph = ephemeris_batch[b] if isinstance(ephemeris_batch, list) else ephemeris_batch

        # H is a constant w.r.t. radar data → torch.from_numpy has no grad_fn.
        # Multiplying X[b] (which has grad_fn) by H keeps the gradient path open.
        H_np = compute_azimuth_filter(meta, eph, Az, Rg)  # (Az, Rg) complex128
        H = torch.from_numpy(H_np).to(dtype=X.dtype, device=X.device)  # cast to match X

        slc_ffts.append(X[b] * H)  # (Az, Rg) — grad_fn preserved from X[b]

    slc_fft = torch.stack(slc_ffts, dim=0)  # (B, Az, Rg)

    # Azimuth IFFT (differentiable)
    slc_c = torch.fft.ifft(slc_fft, dim=1)  # (B, Az, Rg)

    # Strip azimuth buffer
    slc_c = slc_c[:, buffer_size : buffer_size + Az_core, :]  # (B, Az_core, Rg)

    # Split back to (B, 2, Az_core, Rg) via view_as_real — symmetric with view_as_complex above.
    # view_as_real is a true view op; gradient flows back through the contiguous copy.
    slc_ri = torch.view_as_real(slc_c.contiguous())  # (B, Az_core, Rg, 2)
    slc = slc_ri.permute(0, 3, 1, 2).contiguous()  # (B, 2, Az_core, Rg)

    return slc.to(device)
