from __future__ import annotations

import torch
import torch.nn as nn

from basicsr.models.archs.arch_util import LayerNorm2d


class MultiScaleContextCalibration(nn.Module):
    """Stage-level multi-scale context calibration for NAFNet features."""

    def __init__(
        self,
        channels: int,
        expand: int = 1,
        local_kernel_size: int = 3,
        context_kernel_size: int = 5,
        context_dilation: int = 2,
        gate_reduction: int = 4,
        use_channel_gate: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if expand <= 0:
            raise ValueError("expand must be positive.")
        if local_kernel_size % 2 == 0 or context_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size and context_kernel_size must be odd.")
        if context_dilation <= 0:
            raise ValueError("context_dilation must be positive.")
        if gate_reduction <= 0:
            raise ValueError("gate_reduction must be positive.")

        hidden_channels = channels * expand
        local_padding = local_kernel_size // 2
        context_padding = context_kernel_size // 2

        self.norm = LayerNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=True)

        self.local_dw = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=local_kernel_size,
            padding=local_padding,
            groups=hidden_channels,
            bias=True,
        )
        self.context_dw = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=context_kernel_size,
            padding=context_padding,
            groups=hidden_channels,
            bias=True,
        )
        self.context_dilated_dw = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=context_dilation,
            dilation=context_dilation,
            groups=hidden_channels,
            bias=True,
        )

        if use_channel_gate:
            gate_channels = max(hidden_channels // gate_reduction, 8)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_channels, gate_channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_channels, hidden_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.channel_gate = None

        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_in, context_in = self.proj_in(self.norm(x)).chunk(2, dim=1)

        local = self.local_dw(local_in)
        context = self.context_dw(context_in) + self.context_dilated_dw(context_in)
        fused = local * context

        gate_mean = None
        if self.channel_gate is not None:
            gate = self.channel_gate(fused)
            fused = fused * gate
            gate_mean = gate.mean().detach()

        out = self.proj_out(fused)
        y = x + self.gamma * out

        self._last_aux = {
            "gamma_abs_mean": self.gamma.abs().mean().detach(),
            "local_abs_mean": local.abs().mean().detach(),
            "context_abs_mean": context.abs().mean().detach(),
        }
        if gate_mean is not None:
            self._last_aux["gate_mean"] = gate_mean
        return y

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


MCCM = MultiScaleContextCalibration

__all__ = ["MultiScaleContextCalibration", "MCCM"]
