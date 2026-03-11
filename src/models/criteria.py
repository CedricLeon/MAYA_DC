from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from src.models.components.azimuth_focus import to_complex_tensor


class SARRateDistortionCriterion(nn.Module):
    """Rate-distortion objective for focused SAR reconstructions."""

    def __init__(
        self,
        lmbda: float = 0.01,
        include_rate: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.lmbda = lmbda
        self.include_rate = include_rate
        self.eps = eps

    def _compute_rate(self, likelihoods: Mapping[str, Tensor] | None) -> Tensor:
        if not likelihoods:
            return torch.tensor(0.0)

        rates = []
        for likelihood in likelihoods.values():
            rates.append((-torch.log2(likelihood.clamp_min(self.eps))).mean())

        return torch.stack(rates).sum()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        likelihoods: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        prediction_complex = to_complex_tensor(prediction)
        target_complex = to_complex_tensor(target)

        if prediction_complex.shape != target_complex.shape:
            raise ValueError(
                "Prediction and target must share the same focused shape, got "
                f"{tuple(prediction_complex.shape)} and {tuple(target_complex.shape)}"
            )

        diff = prediction_complex - target_complex
        distortion = diff.abs().square().mean()
        amplitude_mse = (
            prediction_complex.abs() - target_complex.abs()
        ).square().mean()

        phase_delta = torch.angle(prediction_complex) - torch.angle(target_complex)
        phase_coherence = torch.abs(torch.exp(1j * phase_delta).mean())
        rate = self._compute_rate(likelihoods).to(device=distortion.device, dtype=distortion.dtype)

        loss = self.lmbda * distortion
        if self.include_rate:
            loss = loss + rate

        return {
            "loss": loss,
            "distortion": distortion,
            "rate": rate,
            "amplitude_mse": amplitude_mse,
            "phase_coherence": phase_coherence,
        }
