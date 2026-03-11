"""Loss functions for SAR compression."""

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import sarpyx loss functions
try:
    from sarpyx.utils.sar_loss import coherence_loss as sarpyx_coherence_loss

    SARPYX_AVAILABLE = True
except ImportError:
    SARPYX_AVAILABLE = False


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


def kde_histogram_loss(pred: Tensor, target: Tensor, num_bins: int = 100) -> Tensor:
    """KDE-based distribution matching loss.

    Uses Gaussian kernel density estimation to compare the distributions
    of predicted and target magnitudes.

    Args:
        pred: Predicted SAR image (B, C, H, W)
        target: Target SAR image (B, C, H, W)
        num_bins: Number of evaluation points for KDE

    Returns:
        Scalar tensor representing distribution loss
    """
    # Convert to magnitude
    if torch.is_complex(pred):
        pred_mag = pred.abs()
        target_mag = target.abs()
    elif pred.shape[1] == 2:
        pred_mag = torch.sqrt(pred[:, 0, :, :] ** 2 + pred[:, 1, :, :] ** 2 + 1e-8)
        target_mag = torch.sqrt(target[:, 0, :, :] ** 2 + target[:, 1, :, :] ** 2 + 1e-8)
    else:
        pred_mag = pred.abs()
        target_mag = target.abs()

    # Flatten
    B = pred_mag.shape[0]
    pred_flat = pred_mag.reshape(B, -1)
    target_flat = target_mag.reshape(B, -1)

    total_loss = torch.tensor(0.0, device=pred.device)

    for b in range(B):
        pred_samples = pred_flat[b]
        target_samples = target_flat[b]

        # Automatic bandwidth selection using Scott's rule
        n_pred = pred_samples.numel()
        n_target = target_samples.numel()
        h_pred = n_pred ** (-1.0 / 5.0) * pred_samples.std()
        h_target = n_target ** (-1.0 / 5.0) * target_samples.std()
        h = (h_pred + h_target) / 2.0

        # Create evaluation grid
        min_val = min(pred_samples.min(), target_samples.min())
        max_val = max(pred_samples.max(), target_samples.max())
        eval_points = torch.linspace(min_val, max_val, num_bins, device=pred.device)

        # Compute KDE for both distributions
        def gaussian_kde(samples, eval_pts, bandwidth):
            # Compute Gaussian kernel for all samples at all evaluation points
            # Shape: (num_samples, num_eval_points)
            diff = samples.unsqueeze(1) - eval_pts.unsqueeze(0)
            kernel = torch.exp(-0.5 * (diff / (bandwidth + 1e-8)) ** 2)
            # Normalize
            density = kernel.sum(dim=0) / (samples.numel() * bandwidth * math.sqrt(2 * math.pi))
            return density

        pred_density = gaussian_kde(pred_samples, eval_points, h)
        target_density = gaussian_kde(target_samples, eval_points, h)

        # L1 distance between densities
        loss_b = torch.abs(pred_density - target_density).mean()
        total_loss += loss_b

    return total_loss / B


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

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Any]:
        """Compute compression loss with compound distortion.

        Args:
            output: Dictionary from compression model containing:
                - "x_hat": Reconstructed data
                - "likelihoods": Dictionary with "y" and "z" likelihoods
            target: Target image

        Returns:
            Dictionary with loss components
        """
        # Rate term
        N, C, H, W = output["x_hat"].shape
        num_pixels = N * H * W

        likelihoods = output.get("likelihoods", {})
        if likelihoods and "y" in likelihoods:
            bpp_y = torch.log(likelihoods["y"]).sum() / (-math.log(2) * num_pixels)
            bpp_z = torch.log(likelihoods["z"]).sum() / (-math.log(2) * num_pixels)
            rate = bpp_y + bpp_z
        else:
            rate = torch.tensor(0.0, device=output["x_hat"].device)

        # Compound distortion term
        distortion_dict = self.compound_loss(output["x_hat"], target)

        # Combined loss
        loss = rate + self.lmbda * distortion_dict["loss"]

        return {
            "loss": loss,
            "rate": rate.item() if isinstance(rate, Tensor) else rate,
            "distortion": distortion_dict["loss"].item(),
            "mse": distortion_dict["mse"],
            "kde": distortion_dict["kde"],
            "coherence": distortion_dict["coherence"],
            "bpp": rate.item() if isinstance(rate, Tensor) else rate,
        }


class CompressionLoss(nn.Module):
    """Compression loss combining rate and distortion.

    Loss = Rate + lambda * Distortion

    where Rate is the expected bitstream length and Distortion measures reconstruction quality.
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        distortion_type: str = "mse",
    ):
        """Initialize compression loss.

        Args:
            lmbda: Rate-distortion tradeoff parameter (higher = more quality)
            distortion_type: Type of distortion metric ("mse", "mae", or "compound")
        """
        super().__init__()
        self.lmbda = lmbda
        self.distortion_type = distortion_type

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Any]:
        """Compute compression loss.

        Args:
            output: Dictionary from compression model containing:
                - "x_hat": Reconstructed RCMC data
                - "likelihoods": Dictionary with "y" and "z" likelihoods
            target: Target SLC image (after azimuth compression)

        Returns:
            Dictionary with loss components
        """
        # Rate loss: negative log-likelihood (bits per pixel)
        N, C, H, W = output["x_hat"].shape
        num_pixels = N * H * W

        # Calculate rate from likelihoods
        likelihoods = output["likelihoods"]
        bpp_y = torch.log(likelihoods["y"]).sum() / (-math.log(2) * num_pixels)
        bpp_z = torch.log(likelihoods["z"]).sum() / (-math.log(2) * num_pixels)
        rate = bpp_y + bpp_z

        # Distortion loss
        if self.distortion_type == "mse":
            distortion = F.mse_loss(output["x_hat"], target)
        elif self.distortion_type == "mae":
            distortion = F.l1_loss(output["x_hat"], target)
        elif self.distortion_type == "compound":
            # Compound loss will be implemented later
            distortion = F.mse_loss(output["x_hat"], target)
        else:
            raise ValueError(f"Unknown distortion type: {self.distortion_type}")

        # Combined loss
        loss = rate + self.lmbda * distortion

        return {
            "loss": loss,
            "rate": rate.item(),
            "distortion": distortion.item(),
            "bpp": (rate.item()),
        }


class RCMCCompressionLoss(nn.Module):
    """Loss for RCMC compression that compares in SLC domain.

    This loss:
    1. Takes compressed/reconstructed RCMC data
    2. Performs azimuth compression to get reconstructed SLC
    3. Compares with target SLC
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        distortion_type: str = "mse",
        azimuth_compression_fn: Any = None,
    ):
        """Initialize RCMC compression loss.

        Args:
            lmbda: Rate-distortion tradeoff
            distortion_type: Distortion metric type
            azimuth_compression_fn: Function to perform azimuth compression
        """
        super().__init__()
        self.lmbda = lmbda
        self.distortion_type = distortion_type
        self.azimuth_compression_fn = azimuth_compression_fn

    def forward(
        self,
        output: Dict[str, Tensor],
        target_slc: Tensor,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute loss in SLC domain.

        Args:
            output: Dictionary from compression model with "x_hat" (reconstructed RCMC)
            target_slc: Ground truth SLC image
            metadata: SAR metadata needed for azimuth compression

        Returns:
            Dictionary with loss components
        """
        import math

        # Get reconstructed RCMC with buffer
        rcmc_recon = output["x_hat"]

        # Perform azimuth compression on reconstructed RCMC
        if self.azimuth_compression_fn is not None:
            slc_recon = self.azimuth_compression_fn(rcmc_recon, metadata)
        else:
            # Placeholder: simple azimuth FFT (not correct but avoids crash)
            slc_recon = torch.fft.fft(rcmc_recon, dim=2).abs()

        # Rate from likelihoods
        N, C, H, W = rcmc_recon.shape
        num_pixels = N * H * W

        likelihoods = output.get("likelihoods", {})
        if likelihoods:
            bpp_y = torch.log(likelihoods["y"]).sum() / (-math.log(2) * num_pixels)
            bpp_z = torch.log(likelihoods["z"]).sum() / (-math.log(2) * num_pixels)
            rate = bpp_y + bpp_z
        else:
            rate = torch.tensor(0.0, device=rcmc_recon.device)

        # Distortion in SLC domain
        if self.distortion_type == "mse":
            distortion = F.mse_loss(slc_recon, target_slc)
        elif self.distortion_type == "mae":
            distortion = F.l1_loss(slc_recon, target_slc)
        else:
            distortion = F.mse_loss(slc_recon, target_slc)

        loss = rate + self.lmbda * distortion

        return {
            "loss": loss,
            "rate": rate.item() if isinstance(rate, Tensor) else rate,
            "distortion": distortion.item(),
            "bpp": rate.item() if isinstance(rate, Tensor) else rate,
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

    def forward(self, output: Dict[str, Tensor], target: Tensor) -> Dict[str, Any]:
        """Compute simple MSE loss.

        Args:
            output: Dictionary with "x_hat" (reconstruction)
            target: Target tensor

        Returns:
            Dictionary with loss components
        """
        import math

        # MSE between reconstruction and target
        mse = F.mse_loss(output["x_hat"], target)

        # Rate from likelihoods if available
        likelihoods = output.get("likelihoods", {})
        if likelihoods and "y" in likelihoods:
            N, C, H, W = output["x_hat"].shape
            num_pixels = N * H * W
            bpp_y = torch.log(likelihoods["y"]).sum() / (-math.log(2) * num_pixels)
            bpp_z = torch.log(likelihoods["z"]).sum() / (-math.log(2) * num_pixels)
            rate = bpp_y + bpp_z
        else:
            rate = torch.tensor(0.0, device=output["x_hat"].device)

        loss = rate + self.lmbda * mse

        return {
            "loss": loss,
            "rate": rate.item() if isinstance(rate, Tensor) else rate,
            "distortion": mse.item(),
            "mse": mse.item(),
            "bpp": rate.item() if isinstance(rate, Tensor) else rate,
        }
