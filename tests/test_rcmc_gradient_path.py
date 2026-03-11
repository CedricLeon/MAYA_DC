from __future__ import annotations

from functools import partial

import numpy as np
import torch

from src.models.components.azimuth_focus import AzimuthFocusMetadata, TorchAzimuthFocus, to_complex_tensor
from src.models.criteria import SARRateDistortionCriterion
from src.models.rcmc_compress_module import RCMCDCmodule


class DummyCompressionNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
        self.latents = torch.nn.Parameter(torch.zeros(1))
        self.quantiles = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_hat = self.proj(x)
        likelihood = torch.sigmoid(self.latents).expand_as(x_hat[:, :1, ...])
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": likelihood,
                "z": torch.sigmoid(self.quantiles).view(1, 1, 1, 1).expand_as(likelihood),
            },
        }

    def aux_loss(self) -> torch.Tensor:
        return self.quantiles.square().sum()


def _build_focus_metadata(
    azimuth_size: int = 6,
    range_size: int = 4,
    crop_azimuth: tuple[int, int] = (0, 0),
    crop_range: tuple[int, int] = (0, 0),
) -> AzimuthFocusMetadata:
    return AzimuthFocusMetadata(
        slant_range_vec=torch.linspace(800.0, 900.0, range_size, dtype=torch.float64),
        azimuth_freq_hz=torch.linspace(-250.0, 250.0, azimuth_size, dtype=torch.float64),
        effective_velocity=torch.linspace(7200.0, 7300.0, azimuth_size, dtype=torch.float64),
        wavelength_m=0.031,
        crop_azimuth=crop_azimuth,
        crop_range=crop_range,
    )


def test_torch_azimuth_focus_matches_reference_formula() -> None:
    focus = TorchAzimuthFocus()
    metadata = _build_focus_metadata(azimuth_size=4, range_size=3)

    x = torch.tensor(
        [
            [
                [[1.0, 0.5, -1.0], [0.0, 1.0, 2.0], [1.0, -1.0, 0.5], [2.0, 0.0, 1.0]],
                [[0.5, -0.5, 0.25], [1.0, 0.0, -1.0], [0.0, 0.25, -0.25], [1.5, -0.5, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    focused = focus(x, metadata).detach().cpu().numpy()
    complex_input = x[:, 0].numpy() + 1j * x[:, 1].numpy()

    az_freq = metadata.azimuth_freq_hz.numpy()
    slant_range = metadata.slant_range_vec.numpy()
    velocity = metadata.effective_velocity.numpy()
    d = np.sqrt(
        np.clip(
            1.0 - ((metadata.wavelength_m**2) * np.square(az_freq)) / (4.0 * np.square(velocity)),
            1e-12,
            None,
        )
    )
    azimuth_filter = np.exp(1j * (4.0 * np.pi * d[:, None] * slant_range[None, :] / metadata.wavelength_m))
    expected = np.fft.ifft(np.fft.fft(complex_input, axis=-2) * azimuth_filter[None, ...], axis=-2)

    assert np.allclose(focused, expected, atol=1e-5)


def test_torch_azimuth_focus_preserves_gradients_with_cropping() -> None:
    focus = TorchAzimuthFocus()
    metadata = _build_focus_metadata(crop_azimuth=(1, 1), crop_range=(1, 0))
    x = torch.randn(2, 2, 6, 4, requires_grad=True)

    focused = focus(x, metadata)
    loss = focused.abs().mean()
    loss.backward()

    assert focused.shape == (2, 4, 3)
    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0


def test_to_complex_tensor_supports_unbatched_channel_first_inputs() -> None:
    x = torch.randn(2, 6, 4)
    complex_x = to_complex_tensor(x)

    assert complex_x.shape == (6, 4)
    assert torch.is_complex(complex_x)


def test_sar_rate_distortion_criterion_returns_differentiable_loss() -> None:
    criterion = SARRateDistortionCriterion(lmbda=0.5)
    prediction = torch.randn(2, 2, 6, 4, requires_grad=True)
    target = torch.randn(2, 2, 6, 4)
    likelihoods = {"y": torch.full((2, 1, 6, 4), 0.75)}

    outputs = criterion(prediction=prediction, target=target, likelihoods=likelihoods)
    outputs["loss"].backward()

    assert outputs["loss"].requires_grad
    assert prediction.grad is not None
    assert set(outputs) == {"loss", "distortion", "rate", "amplitude_mse", "phase_coherence"}


def test_rcmc_module_model_step_keeps_gradients_and_updates_weights() -> None:
    focus = TorchAzimuthFocus()
    metadata = _build_focus_metadata()
    target_crop = focus(torch.randn(2, 2, 6, 4), metadata).detach()

    module = RCMCDCmodule(
        net=DummyCompressionNet(),
        criterion=SARRateDistortionCriterion(lmbda=0.2),
        focus=focus,
        net_optimizer=partial(torch.optim.SGD, lr=0.1),
        aux_optimizer=partial(torch.optim.SGD, lr=0.1),
        scheduler=None,
    )

    batch = {
        "rcmc_input": torch.randn(2, 2, 6, 4),
        "slc_target": target_crop,
        "focus_metadata": metadata,
    }

    criterion, aux_loss = module.model_step(batch)
    net_optimizer, aux_optimizer = module.configure_optimizers()

    before = module.net.proj.weight.detach().clone()
    net_optimizer.zero_grad()
    aux_optimizer.zero_grad()

    criterion["loss"].backward(retain_graph=True)
    assert criterion["loss"].requires_grad
    assert module.net.proj.weight.grad is not None

    net_optimizer.step()
    aux_loss.backward()
    aux_optimizer.step()

    after = module.net.proj.weight.detach()
    assert not torch.allclose(before, after)
