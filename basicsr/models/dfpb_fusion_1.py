from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d
from .dfpb import DualFrequencyProgressiveBlock
from .wavedfpb import WaveletDualFrequencyProgressiveBlock


class DualFrequencySkipFusion(nn.Module):
    """
    同尺度跳连融合模块（DFPB 风格细化核心）。

    主要用途：解码器与编码器同尺度特征融合。
    - x_dec: 解码器上采样后的特征。
    - x_skip: 编码器同尺度的跳连特征。

    设计要点：
    1) 先做原始 skip 相加，保证初始化等价于 NAFNet 传统融合。
    2) 再通过门控融合 + 频域细化模块增强融合特征。
    """

    def __init__(
        self,
        channels: int,
        block_type: str = "wavelet",
        reduction: int = 4,
        dfpb_kwargs: dict | None = None,
    ) -> None:
        """初始化模块与子结构。"""
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive.")

        self.channels = channels
        self.block_type = block_type
        # 解码器/跳连特征先做归一化与1x1对齐到同一通道空间。
        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.dec_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # 门控支路用于控制 skip 注入强度。
        self.skip_gate = nn.Sequential(
            nn.Conv2d(channels * 2, max(channels // reduction, 8), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // reduction, 8), channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        dfpb_kwargs = {} if dfpb_kwargs is None else dict(dfpb_kwargs)
        # 频域细化核心选择：DFPB 或 Wavelet-DFPB。
        if block_type == "dfpb":
            self.frequency_block = DualFrequencyProgressiveBlock(channels=channels, **dfpb_kwargs)
        elif block_type in {"wavelet", "wavedfpb", "wavelet_dfpb"}:
            dfpb_kwargs.pop("low_kernel_size", None)
            self.frequency_block = WaveletDualFrequencyProgressiveBlock(channels=channels, **dfpb_kwargs)
        else:
            raise ValueError(f"Unknown block_type: {block_type}.")

        # 输出投影 + 残差缩放，gamma 初始化为0以保持初始等价于 x_dec + x_skip。
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        前向流程（从输入到输出）：
        1) x_dec 与 x_skip 形状检查。
        2) base = x_dec + x_skip（保留原始 skip 行为）。
        3) 对 x_dec/x_skip 归一化 + 1x1 投影。
        4) 通过门控融合得到 mixed。
        5) 频域细化（DFPB 或 Wavelet-DFPB）。
        6) 残差回加得到最终输出。
        """
        if x_dec.shape != x_skip.shape:
            raise ValueError(
                "DualFrequencySkipFusion expects same-scale, same-channel features; "
                f"got x_dec={tuple(x_dec.shape)} and x_skip={tuple(x_skip.shape)}."
            )

        # 先做原始 skip 相加，再做门控融合与频域细化。
        base = x_dec + x_skip
        # 对解码器特征做 LayerNorm + 1x1 通道对齐。
        dec = self.dec_proj(self.norm_dec(x_dec))
        # 对跳连特征做 LayerNorm + 1x1 通道对齐。
        skip = self.skip_proj(self.norm_skip(x_skip))
        # 拼接后通过门控网络生成融合权重（0~1）。
        gate = self.skip_gate(torch.cat([dec, skip], dim=1))
        # 门控融合：保留 dec 主干，引入 gated skip。
        mixed = dec + gate * skip
        # 频域细化模块。
        refined = self.frequency_block(mixed)
        # 输出：base 残差 + 细化结果（带 gamma 缩放）。
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


DualFrequencySkipFusionGated = DualFrequencySkipFusion

__all__ = ["DualFrequencySkipFusion", "DualFrequencySkipFusionGated"]
