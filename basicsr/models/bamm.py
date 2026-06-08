from __future__ import annotations

import torch
import torch.nn as nn

from basicsr.models.archs.arch_util import LayerNorm2d


class BlurAwareModulation(nn.Module):
    """Lightweight blur-aware feature modulation with directional blur cues."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        branch_reduction: int = 4,
        kernel_size: int = 7,
        dilation: int = 2,
        use_spatial_gate: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if branch_reduction <= 0:
            raise ValueError("branch_reduction must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")

        hidden_channels = max(channels // reduction, 8)
        branch_channels = max(channels // branch_reduction, 16)
        padding = kernel_size // 2

        self.norm = LayerNorm2d(channels)
        self.blur_reduce = nn.Conv2d(channels, branch_channels, kernel_size=1, bias=True)
        self.dw_horizontal = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=branch_channels,
            bias=True,
        )
        self.dw_vertical = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=branch_channels,
            bias=True,
        )
        self.dw_dilated = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=branch_channels,
            bias=True,
        )
        self.blur_proj = nn.Conv2d(branch_channels, channels, kernel_size=1, bias=True)

        self.channel_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1, bias=True),
        )

        if use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blur_reduce(self.norm(x))
        blur = self.dw_horizontal(z) + self.dw_vertical(z) + self.dw_dilated(z)
        blur = self.blur_proj(blur)

        scale, shift = self.channel_mod(blur).chunk(2, dim=1)
        modulated = x * (1.0 + scale) + shift

        gate_mean = None
        if self.spatial_gate is not None:
            gate = self.spatial_gate(blur)
            modulated = modulated * gate
            gate_mean = gate.mean().detach()

        out = self.out_proj(modulated)
        y = x + self.gamma * out

        self._last_aux = {
            "gamma_abs_mean": self.gamma.abs().mean().detach(),
            "blur_abs_mean": blur.abs().mean().detach(),
            "scale_abs_mean": scale.abs().mean().detach(),
            "shift_abs_mean": shift.abs().mean().detach(),
        }
        if gate_mean is not None:
            self._last_aux["spatial_gate_mean"] = gate_mean
        return y

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


BAMM = BlurAwareModulation

__all__ = ["BlurAwareModulation", "BAMM"]