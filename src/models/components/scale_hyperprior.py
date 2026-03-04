from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel
from torch import Size, Tensor

# from src.models.components.layers import ResidualBlock, make_activation
# from src.utils.debug import log_tensor_shape


class ResidualScaleHyperprior(CompressionModel):
    """Scale Hyperprior model."""

    def __init__(
        self,
        nb_input_channels: int = 2,
        nb_channels_main: int = 128,
        activation: str = "gdn",
    ):
        super().__init__()
        N = nb_channels_main
        M = 2 * N  # Number of channels for hyperprior
        self.nb_input_channels: int = nb_input_channels
        self.activation: str = activation

        # Big ugly parameter for shape logging during inference
        self.DEBUG_MODE: bool = False

        self.entropy_bottleneck: EntropyBottleneck = EntropyBottleneck(N)
        self.gaussian_conditional: GaussianConditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            nn.Conv2d(self.nb_input_channels, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=1, output_padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, output_padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, output_padding=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(
                N, self.nb_input_channels, kernel_size=5, stride=2, padding=1, output_padding=2
            ),
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, output_padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=1, output_padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        """Complete forward pass through the model, returning the reconstructed output and
        likelihoods."""
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
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

    def decompress(self, strings, shape):
        """Decompress the given `strings` using the provided `shape` information, returning the
        reconstructed tensor."""
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
