from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d
from .dfpb import DualFrequencyProgressiveBlock
from .wavedfpb import WaveletDualFrequencyProgressiveBlock


class DualFrequencySkipFusionNoDeform(nn.Module):
    """Original-style frequency fusion without deformable alignment."""

    def __init__(
        self,
        channels: int,
        block_type: str = "wavelet",
        reduction: int = 4,
        dfpb_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive.")

        self.channels = channels
        self.block_type = block_type

        # Same as the original fusion path: normalize and project both branches.
        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.dec_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Lightweight gate to control how much skip information is injected.
        self.skip_gate = nn.Sequential(
            nn.Conv2d(channels * 2, max(channels // reduction, 8), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // reduction, 8), channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        dfpb_kwargs = {} if dfpb_kwargs is None else dict(dfpb_kwargs)
        # Explicitly disable deformable fusion inside the frequency block.
        dfpb_kwargs["use_deformable_fusion"] = False

        if block_type == "dfpb":
            self.frequency_block = DualFrequencyProgressiveBlock(channels=channels, **dfpb_kwargs)
        elif block_type in {"wavelet", "wavedfpb", "wavelet_dfpb"}:
            dfpb_kwargs.pop("low_kernel_size", None)
            self.frequency_block = WaveletDualFrequencyProgressiveBlock(channels=channels, **dfpb_kwargs)
        else:
            raise ValueError(f"Unknown block_type: {block_type}.")

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        if x_dec.shape != x_skip.shape:
            raise ValueError(
                "DualFrequencySkipFusionNoDeform expects same-scale, same-channel features; "
                f"got x_dec={tuple(x_dec.shape)} and x_skip={tuple(x_skip.shape)}."
            )

        base = x_dec + x_skip
        dec = self.dec_proj(self.norm_dec(x_dec))
        skip = self.skip_proj(self.norm_skip(x_skip))

        gate = self.skip_gate(torch.cat([dec, skip], dim=1))
        mixed = dec + gate * skip
        refined = self.frequency_block(mixed)
        out = base + self.gamma * self.out_proj(refined)

        self._last_aux = {
            "skip_gate_mean": gate.mean().detach(),
            "skip_gate_std": gate.std().detach(),
            "gamma_abs_mean": self.gamma.abs().mean().detach(),
        }
        if hasattr(self.frequency_block, "get_last_aux"):
            self._last_aux.update(
                {f"frequency_{key}": value for key, value in self.frequency_block.get_last_aux().items()}
            )
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


DualFrequencySkipFusion = DualFrequencySkipFusionNoDeform

__all__ = ["DualFrequencySkipFusionNoDeform", "DualFrequencySkipFusion"]