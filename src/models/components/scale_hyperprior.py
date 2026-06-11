from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
from torch import Size, Tensor


@dataclass
class Likelihoods:
    """Likelihoods for the main latent representation `y` and the hyperprior `z`."""

    y: Tensor
    z: Tensor


@dataclass
class ForwardOutput:
    """Output of the forward pass through the model, containing the reconstructed output and
    likelihoods."""

    x_hat: Tensor
    likelihoods: Likelihoods


def make_activation(act_name: str, channels: int, inverse: bool = False) -> nn.Module:
    """Return an activation module based on the given name."""
    t = act_name.lower()
    if t == "gdn":
        return GDN(channels, inverse=inverse)
    if t == "relu":
        return nn.ReLU(inplace=True)
    if t in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unknown activation type: {act_name}")


class ScaleHyperprior(CompressionModel):
    """Scale Hyperprior model with custom number of input channels (CompressAI models enforce 3).

    args:
        nb_input_channels: Number of channels in the input tensor (default 2 for complex SAR)
        nb_channels_main: Number of channels in the main encoder/decoder (N, default 128)
        nb_channels_latent: Number of channels in the latent space (M, default 2*N).
            In Ballé et al. 2018, M and N are independent.  Set explicitly for ablations.
        activation: Activation function for main encoder/decoder ("gdn", "relu", "identity").
            Hyperprior always uses ReLU regardless of this setting.
    """

    def __init__(
        self,
        nb_input_channels: int = 2,
        nb_channels_main: int = 128,
        nb_channels_latent: int = 256,
        activation: str = "gdn",
        non_square_kernels: bool = False,
    ):
        super().__init__()
        N = nb_channels_main
        M = nb_channels_latent
        self.nb_input_channels: int = nb_input_channels
        self.nb_channels_latent: int = nb_channels_latent
        self.activation: str = activation
        self.non_square_kernels: bool = non_square_kernels

        # Non-square kernel helpers.
        # Tensors have shape (B, C, Az, Rg), so H = azimuth, W = range.
        # When non_square_kernels=True, the azimuth (H) dimension of every kernel is
        # scaled by _NSK_FACTOR relative to the range (W) dimension.
        # Factor 3 matches the 3:1 Az:Rg aspect ratio of the default patch
        # (Az + 2*buf : Rg = 1536 : 512), giving the network more azimuth context per step.
        # Padding is kept 'same' for stride-1 layers and halved-output for stride-2 layers.
        _NSK_FACTOR = 3

        def ks(k: int) -> int | tuple[int, int]:
            """Return kernel size: k (square) or (k*factor, k) i.e. (kH_az, kW_rg) (non-square)."""
            return (k * _NSK_FACTOR, k) if non_square_kernels else k

        def pd(k: int) -> int | tuple[int, int]:
            """Return 'same' padding: scalar (square) or (pH_az, pW_rg) (non-square)."""
            return (
                ((k * _NSK_FACTOR - 1) // 2, (k - 1) // 2) if non_square_kernels else (k - 1) // 2
            )

        self.entropy_bottleneck: EntropyBottleneck = EntropyBottleneck(N)
        self.gaussian_conditional: GaussianConditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            nn.Conv2d(self.nb_input_channels, N, kernel_size=ks(5), stride=2, padding=pd(5)),
            make_activation(self.activation, N),
            nn.Conv2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5)),
            make_activation(self.activation, N),
            nn.Conv2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5)),
            make_activation(self.activation, N),
            nn.Conv2d(N, M, kernel_size=ks(5), stride=2, padding=pd(5)),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=ks(5), stride=2, padding=pd(5), output_padding=1),
            make_activation(self.activation, N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5), output_padding=1),
            make_activation(self.activation, N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5), output_padding=1),
            make_activation(self.activation, N, inverse=True),
            nn.ConvTranspose2d(
                N,
                self.nb_input_channels,
                kernel_size=ks(5),
                stride=2,
                padding=pd(5),
                output_padding=1,
            ),
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=ks(3), stride=1, padding=pd(3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5)),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5), output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=ks(5), stride=2, padding=pd(5), output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M, kernel_size=ks(3), stride=1, padding=pd(3)),
            nn.ReLU(inplace=True),
        )

    @property
    def main_downsampling_factor(self) -> int:
        """Return the overall downsampling factor of the main encoder."""
        return 16  # 2^4 from the 4 stride=2 layers in g_a

    @property
    def hyper_downsampling_factor(self) -> int:
        """Return the overall downsampling factor of the hyperprior."""
        return 8  # 2^3 from the 3 stride=2 layers in h_a

    def forward(self, x: Tensor) -> ForwardOutput:
        """Complete forward pass through the model, returning the reconstructed output and
        likelihoods."""
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        return ForwardOutput(
            x_hat=x_hat, likelihoods=Likelihoods(y=y_likelihoods, z=z_likelihoods)
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from ``state_dict``.

        Reconstructs the constructor arguments from the weight tensor shapes:
        - ``g_a.0.weight`` shape is ``(N, nb_input_channels, kH, kW)``
          → ``nb_input_channels = size(1)``, ``nb_channels_main N = size(0)``
        - ``g_a.6.weight`` shape is ``(M, N, kH, kW)``
          → ``nb_channels_latent M = size(0)``
        - Activation detected by presence of GDN parameter ``g_a.1._beta``.
        - Non-square kernels detected by ``kW != kH`` in ``g_a.0.weight``.
        """
        nb_input_channels = state_dict["g_a.0.weight"].size(1)
        nb_channels_main = state_dict["g_a.0.weight"].size(0)  # N
        nb_channels_latent = state_dict["g_a.6.weight"].size(0)  # M
        activation = "gdn" if "g_a.1._beta" in state_dict else "relu"
        non_square_kernels = state_dict["g_a.0.weight"].size(-1) != state_dict[
            "g_a.0.weight"
        ].size(-2)
        net = cls(
            nb_input_channels, nb_channels_main, nb_channels_latent, activation, non_square_kernels
        )
        net.load_state_dict(state_dict)
        return net

    def compress(self, x: Tensor) -> dict[str, Any]:
        """Compress an input tensor `x` into a dictionary containing the compressed bitstrings and
        shape information."""
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings: Any, shape: tuple[int, int]) -> dict[str, Tensor]:
        """Decompress the given `strings` using the provided `shape` information, returning the
        reconstructed tensor."""
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat)  # .clamp_(0, 1)
        return {"x_hat": x_hat}
