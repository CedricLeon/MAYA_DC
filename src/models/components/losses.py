"""Loss functions and quality metrics for SAR compression."""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from src.models.components.scale_hyperprior import ForwardOutput, Likelihoods
from src.utils.processing_utils import EPS, phys_to_linA_torch

# ---------------------------------------------------------------------------
# Quality metrics (no gradient required; used at validation / test time)
# ---------------------------------------------------------------------------


def complex_correlation_metric(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Per-image complex correlation magnitude (coherence), returned as ``(mean, std)``.

    For each image in the batch, computes
    :math:`|\\langle\\hat{s}, s^*\\rangle| / (\\|\\hat{s}\\| \\cdot \\|s\\|)`.
    This is the single global coherence per patch — the primary SAR quality
    indicator (F2).

    Args:
        pred:   ``(B, 2, H, W)`` float — real/imag channels.
        target: ``(B, 2, H, W)`` float — real/imag channels.

    Returns:
        mean: batch-mean coherence in [0, 1].
        std:  batch-std coherence.
    """
    # (B, 2, H, W) → (B, H, W) complex
    pred_c = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())  # (B, H, W)
    target_c = torch.view_as_complex(target.permute(0, 2, 3, 1).contiguous())  # (B, H, W)

    B = pred_c.shape[0]
    pf = pred_c.reshape(B, -1)  # (B, N)
    tf = target_c.reshape(B, -1)  # (B, N)

    # |<pred, target*>| / (||pred|| * ||target||) per image
    num = (pf * tf.conj()).sum(dim=1).abs()  # (B,)
    denom = pf.abs().pow(2).sum(dim=1).sqrt() * tf.abs().pow(2).sum(dim=1).sqrt() + 1e-8  # (B,)
    gamma = num / denom  # (B,), in [0, 1]

    return gamma.mean(), gamma.std()


def psnr_amplitude(pred: Tensor, target: Tensor, eps: float = EPS) -> Tensor:
    """Peak Signal-to-Noise Ratio on amplitude images, averaged over the batch.

    Each image uses its own ``data_range = max(|target|)`` so the metric is
    scale-invariant to different patches (F3).

    .. note::
        Both tensors are expected to be in the same normalised domain
        (e.g. after ``minmax_normalize``). Comparing across different
        normalisation schemes will give meaningless values.

    Args:
        pred:   ``(B, 2, H, W)`` float — real/imag channels.
        target: ``(B, 2, H, W)`` float — real/imag channels.
        eps:    Small constant to avoid log(0).

    Returns:
        Scalar mean PSNR [dB] over the batch.
    """
    pred_linA = phys_to_linA_torch(pred)  # (B, H, W)
    target_linA = phys_to_linA_torch(target)  # (B, H, W)

    B = pred_linA.shape[0]
    data_range = target_linA.reshape(B, -1).amax(dim=1)  # (B,)
    mse_per_image = (
        F.mse_loss(pred_linA, target_linA, reduction="none").reshape(B, -1).mean(dim=1)
    )  # (B,)
    psnr = 10.0 * torch.log10(data_range**2 / (mse_per_image + eps))  # (B,)
    return psnr.mean()


def ssim_amplitude(pred: Tensor, target: Tensor) -> Tensor:
    """Structural Similarity Index on amplitude images, averaged over the batch (F3).

    Uses the torchmetrics SSIM implementation.  ``data_range`` is fixed to
    ``√2`` — the theoretical maximum amplitude when both channels are
    normalised to ``[0, 1]`` (real² + imag² ≤ 2).  Using a fixed value
    makes SSIM scores comparable across batches and epochs; a per-batch
    adaptive ``amax`` would shift the stability constants and produce
    incomparable values.

    .. note::
        Both inputs must be in the same normalised domain, e.g. after
        ``minmax_normalize(., GT_MIN, GT_MAX)`` so channels are in ``[0, 1]``.

    Args:
        pred:   ``(B, 2, H, W)`` float — real/imag channels.
        target: ``(B, 2, H, W)`` float — real/imag channels.
        eps:    Regulariser added before taking the square root of the amplitude.

    Returns:
        Scalar mean SSIM in [0, 1] over the batch.
    """
    pred_linA = phys_to_linA_torch(pred).unsqueeze(1)  # (B,1,H,W)
    target_linA = phys_to_linA_torch(target).unsqueeze(1)  # (B,1,H,W)

    # Fixed data_range: channels normalised to [0,1] ⇒ max amplitude = sqrt(2).
    data_range = math.sqrt(2.0)
    result = ssim_fn(pred_linA, target_linA, data_range=data_range, return_full_image=False)
    return result if isinstance(result, Tensor) else result[0]


def phase_preservation_metric(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Per-image phase preservation metric, returned as ``(mean, std)`` (F9).

    Measures how well the reconstructed phase matches the target, *ignoring
    amplitude*.  For each pixel, the phasor difference
    :math:`\\exp(j(\\phi_{\\hat{s}} - \\phi_s))` is computed on the unit circle.
    The metric is:

    .. math::

        \\gamma_\\phi = 1 - \\left|\\overline{\\exp(j(\\phi_{\\hat{s}} - \\phi_s))}\\right|

    where the bar denotes the spatial mean over the patch.
    A value of **0** = perfect phase preservation; **1** = fully random phase.

    This is complementary to ``complex_correlation_metric``, which conflates
    phase and amplitude.  Use this metric to diagnose phase-only degradation.

    Args:
        pred:   ``(B, 2, H, W)`` float — real/imag channels.
        target: ``(B, 2, H, W)`` float — real/imag channels.

    Returns:
        mean: batch-mean phase error in [0, 1].
        std:  batch-std phase error.
    """
    pred_c = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())  # (B, H, W)
    target_c = torch.view_as_complex(target.permute(0, 2, 3, 1).contiguous())  # (B, H, W)

    # Project to unit circle to isolate phase
    pred_ph = pred_c / (pred_c.abs() + 1e-8)  # exp(j*phi_pred)
    target_ph = target_c / (target_c.abs() + 1e-8)  # exp(j*phi_target)

    # Phase-difference phasor: exp(j*(phi_pred - phi_target))
    diff_ph = pred_ph * target_ph.conj()  # (B, H, W)

    B = diff_ph.shape[0]
    gamma = 1.0 - diff_ph.reshape(B, -1).mean(dim=1).abs()  # (B,), in [0, 1]
    return gamma.mean(), gamma.std()


def complex_coherence_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute interferometric coherence loss.

    Loss = 1 - coherence, where coherence is the magnitude of complex correlation.

    Args:
        pred: Predicted complex SAR image (real channels or complex)
        target: Target complex SAR image (real channels or complex)

    Returns:
        Scalar tensor representing coherence loss
    """
    # Convert real/imag channels to complex if needed
    if not torch.is_complex(pred) and pred.shape[1] == 2:
        pred_complex = torch.complex(pred[:, 0, :, :], pred[:, 1, :, :])
        target_complex = torch.complex(target[:, 0, :, :], target[:, 1, :, :])
    else:
        pred_complex = pred
        target_complex = target

    # Flatten for correlation
    pred_flat = pred_complex.flatten()
    target_flat = target_complex.flatten()

    # Compute coherence
    numerator = torch.abs((pred_flat * torch.conj(target_flat)).mean())
    denom = torch.sqrt((torch.abs(pred_flat) ** 2).mean() * (torch.abs(target_flat) ** 2).mean())

    coherence = numerator / (denom + 1e-8)
    return 1.0 - coherence


def _gaussian_kde(samples: Tensor, eval_points: Tensor, bandwidth: Tensor) -> Tensor:
    """Differentiable Gaussian kernel density estimate.

    Args:
        samples:     Flat sample values ``[N]``.
        eval_points: Grid to evaluate the KDE on ``[M]``.
        bandwidth:   KDE bandwidth (scalar tensor).

    Returns:
        Density estimates at ``eval_points``, shape ``[M]``.
    """
    ep = eval_points.view(-1, 1)  # [M, 1]
    s = samples.view(1, -1)  # [1, N]

    z = ((ep - s) / bandwidth) ** 2 / 2.0  # [M, N]

    gaussian_const = 1.0 / torch.sqrt(
        torch.tensor(2.0 * math.pi, dtype=ep.dtype, device=ep.device)
    )
    kernel = gaussian_const * torch.exp(-z)  # [M, N]

    return kernel.sum(dim=1) / (s.shape[1] * bandwidth)  # [M]


def kde_histogram_loss(pred: Tensor, target: Tensor, num_bins: int = 100) -> Tensor:
    """Differentiable KDE-based distribution matching loss.

    Computes the L1 distance between the Gaussian KDEs of ``pred`` and
    ``target`` magnitudes using Scott's bandwidth rule.  All operations are
    PyTorch-native, so gradients flow through the KDE computation.

    Ported from ``MultiDomainSARLoss._histogram_l1_loss`` in
    ``srp/sarpyx/utils/losses.py``.

    Args:
        pred:     Predicted SAR image ``(B, 2, H, W)`` real/imag channels.
        target:   Target SAR image ``(B, 2, H, W)``.
        num_bins: Number of evaluation points for the KDE (default 100).

    Returns:
        Scalar L1 distance between KDEs, averaged over the batch.
    """
    # Convert to linear amplitude
    if torch.is_complex(pred):
        pred_linA = pred.abs()
        target_linA = target.abs()
    elif pred.shape[1] == 2:
        pred_linA = phys_to_linA_torch(pred)
        target_linA = phys_to_linA_torch(target)
    else:
        pred_linA = pred.abs()
        target_linA = target.abs()

    B = pred_linA.shape[0]
    pred_flat = pred_linA.view(B, -1)
    target_flat = target_linA.view(B, -1)

    total_loss = torch.tensor(0.0, device=pred.device)

    for b in range(B):
        ps = pred_flat[b]
        ts = target_flat[b]

        # Scott's rule: h = std * n^(-1/5)
        h_p = (ps.std() + 1e-8) * (ps.numel() ** (-0.2))
        h_t = (ts.std() + 1e-8) * (ts.numel() ** (-0.2))
        h = (h_p + h_t) / 2.0

        # Evaluation grid covering both distributions + 10 % margin
        lo = torch.min(ps.min(), ts.min())
        hi = torch.max(ps.max(), ts.max())
        margin = (hi - lo) * 0.1
        eval_pts = torch.linspace(
            (lo - margin).item(),
            (hi + margin).item(),
            num_bins,
            device=pred.device,
        )

        pred_kde = _gaussian_kde(ps, eval_pts, h)
        target_kde = _gaussian_kde(ts, eval_pts, h)

        # Approximate L1 integral via trapezoidal sum.
        # dx is a bin-width constant — detach so it does not contribute spurious
        # gradients through hi/lo (which depend on ps.max() and ps.min()).
        dx = ((hi - lo + 2.0 * margin) / num_bins).detach()
        total_loss = total_loss + torch.abs(pred_kde - target_kde).sum() * dx

    return total_loss / B


def estimate_rate(likelihoods: Likelihoods, compress_shape: Tuple[int, int, int, int]) -> Tensor:
    """Compute the bitrate (rate) from the likelihoods estimnated by the model."""
    N, _C, H, W = compress_shape
    num_pixels = N * H * W

    bpp_y = torch.log(likelihoods.y).sum() / (-math.log(2) * num_pixels)
    bpp_z = torch.log(likelihoods.z).sum() / (-math.log(2) * num_pixels)
    return bpp_y + bpp_z


class CompoundSARLoss(nn.Module):
    """Compound loss for SAR compression combining MSE, KDE, and phase coherence.

    Loss = MSE + delta_kde * KDE + delta_coherence * Coherence

    This specialized loss is designed for SAR data to:
    1. MSE: Minimize average pixel-wise error
    2. KDE: Match the overall distribution of magnitudes (preserves texture/speckle)
    3. Coherence: Preserve phase structure (critical for interferometry)

    Args:
        delta_kde: Weight for KDE distribution matching term
        delta_coherence: Weight for phase coherence term
        num_kde_bins: Number of evaluation points for KDE
    """

    def __init__(
        self,
        delta_kde: float = 0.1,
        delta_coherence: float = 0.1,
        num_kde_bins: int = 100,
    ):
        """Initialize compound SAR loss."""
        super().__init__()
        self.delta_kde = delta_kde
        self.delta_coherence = delta_coherence
        self.num_kde_bins = num_kde_bins

    def forward(self, pred: Tensor, target: Tensor) -> Dict[str, Any]:
        """Compute compound loss.

        Args:
            pred: Predicted SAR image (B, C, H, W) where C=2 for real/imag
            target: Target SAR image (B, C, H, W)

        Returns:
            Dictionary with loss components
        """
        # 1. MSE loss
        mse = F.mse_loss(pred, target)

        # 2. KDE distribution matching
        kde_loss = kde_histogram_loss(pred, target, num_bins=self.num_kde_bins)

        # 3. Phase coherence
        coherence = complex_coherence_loss(pred, target)

        # Combined loss
        total_loss = mse + self.delta_kde * kde_loss + self.delta_coherence * coherence

        return {
            "loss": total_loss,
            "mse": mse.item(),
            "kde": kde_loss.item(),
            "coherence": coherence.item(),
        }


class CompoundCompressionLoss(nn.Module):
    """Compression loss with compound SAR distortion.

    Combines rate term with compound distortion (MSE + KDE + Coherence).

    Loss = Rate + lambda * (MSE + delta_kde * KDE + delta_coherence * Coherence)
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        delta_kde: float = 0.1,
        delta_coherence: float = 0.1,
        num_kde_bins: int = 100,
        patch_size: Tuple[int, int] = (512, 512),
        azimuth_buffer: int = 512,
    ):
        """Initialize compound compression loss.

        Args:
            lmbda: Rate-distortion tradeoff parameter
            delta_kde: Weight for KDE term within distortion
            delta_coherence: Weight for coherence term within distortion
            num_kde_bins: Number of bins for KDE evaluation
        """
        super().__init__()
        self.lmbda = lmbda
        self.compound_loss = CompoundSARLoss(delta_kde, delta_coherence, num_kde_bins)
        self.slc_shape = patch_size
        self.az_buffer = azimuth_buffer
        self.rcmc_shape = (patch_size[0] + 2 * azimuth_buffer, patch_size[1])

    def forward(self, output: ForwardOutput, target: Tensor) -> Dict[str, Any]:
        """Compute compression loss with compound distortion that returns a dictionary of
        components."""
        # recon_shape is the batch from x_hat, the channels from x_hat and rcmc_shape height and width
        recon_shape = (output.x_hat.shape[0], output.x_hat.shape[1], *self.rcmc_shape)
        rate = estimate_rate(output.likelihoods, recon_shape)
        distortion_dict = self.compound_loss(output.x_hat, target)

        loss = rate + self.lmbda * distortion_dict["loss"]

        return {
            "loss": loss,
            "rate": rate.item(),
            "distortion": distortion_dict["loss"].item(),
            "mse": distortion_dict["mse"],
            "kde": distortion_dict["kde"],
            "coherence": distortion_dict["coherence"],
        }


class SimpleMSELoss(nn.Module):
    """Simple MSE loss for quick testing.

    Compares reconstructed RCMC directly with target without azimuth compression.
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        patch_size: Tuple[int, int] = (512, 512),
        azimuth_buffer: int = 512,
    ):
        """Initialize simple MSE loss.

        Args:
            lmbda: Rate-distortion tradeoff
        """
        super().__init__()
        self.lmbda = lmbda
        self.slc_shape = patch_size
        self.az_buffer = azimuth_buffer
        self.rcmc_shape = (patch_size[0] + 2 * azimuth_buffer, patch_size[1])

    def forward(self, output: ForwardOutput, target: Tensor) -> Dict[str, Any]:
        """Compute simple MSE loss."""
        recon_shape = (output.x_hat.shape[0], output.x_hat.shape[1], *self.rcmc_shape)
        rate = estimate_rate(output.likelihoods, recon_shape)
        mse = F.mse_loss(output.x_hat, target)

        loss = rate + self.lmbda * mse

        return {
            "loss": loss,
            "rate": rate.item(),
            "distortion": mse.item(),
            "mse": mse.item(),
        }
