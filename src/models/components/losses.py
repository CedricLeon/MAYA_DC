"""Loss functions for SAR compression."""

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.components.scale_hyperprior import ForwardOutput, Likelihoods


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
    # Convert to magnitude
    if torch.is_complex(pred):
        pred_mag = pred.abs()
        target_mag = target.abs()
    elif pred.shape[1] == 2:
        pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
        target_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + 1e-8)
    else:
        pred_mag = pred.abs()
        target_mag = target.abs()

    B = pred_mag.shape[0]
    pred_flat = pred_mag.view(B, -1)
    target_flat = target_mag.view(B, -1)

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

        # Approximate L1 integral via trapezoidal sum
        dx = (hi - lo + 2.0 * margin) / num_bins
        total_loss = total_loss + torch.abs(pred_kde - target_kde).sum() * dx

    return total_loss / B


def estimate_rate(output: ForwardOutput) -> Tensor:
    """Compute the bitrate (rate) from the likelihoods estimnated by the model."""
    N, _C, H, W = output.x_hat.shape
    num_pixels = N * H * W

    likelihoods = output.likelihoods
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

    def forward(self, output: ForwardOutput, target: Tensor) -> Dict[str, Any]:
        """Compute compression loss with compound distortion that returns a dictionary of
        components."""
        rate = estimate_rate(output)
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

    def __init__(self, lmbda: float = 0.01):
        """Initialize simple MSE loss.

        Args:
            lmbda: Rate-distortion tradeoff
        """
        super().__init__()
        self.lmbda = lmbda

    def forward(self, output: ForwardOutput, target: Tensor) -> Dict[str, Any]:
        """Compute simple MSE loss."""
        rate = estimate_rate(output)
        mse = F.mse_loss(output.x_hat, target)

        loss = rate + self.lmbda * mse

        return {
            "loss": loss,
            "rate": rate.item(),
            "distortion": mse.item(),
            "mse": mse.item(),
        }
