from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
except Exception:  # pragma: no cover - optional runtime dependency
    tv_deform_conv2d = None


class LayerNorm2d(nn.Module):
    """按空间位置做 LayerNorm，只在通道维上归一化。"""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConditionedDeformableConv2d(nn.Module):
    """条件可变形卷积：offset 和 mask 都由条件特征预测。"""

    def __init__(
        self,
        channels: int,
        condition_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_deformable: bool = True,
        align_groups: int | str = 1,
        offset_reduction: int = 1,
        max_offset: float | None = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if offset_reduction <= 0:
            raise ValueError("offset_reduction must be positive.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.align_groups = channels if align_groups == "depthwise" else int(align_groups)
        if self.align_groups <= 0 or channels % self.align_groups != 0:
            raise ValueError("align_groups must be positive and divide channels.")
        self.use_deformable = bool(use_deformable and tv_deform_conv2d is not None)
        self.max_offset = None if max_offset is None else float(max_offset)
        if self.max_offset is not None and self.max_offset <= 0:
            raise ValueError("max_offset must be positive when set.")

        self.weight = nn.Parameter(
            torch.empty(channels, channels // self.align_groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if self.use_deformable:
            out_channels = 3 * kernel_size * kernel_size
            if offset_reduction == 1:
                self.offset_mask = nn.Conv2d(
                    condition_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            else:
                hidden_channels = max(condition_channels // offset_reduction, 8)
                self.offset_mask = nn.Sequential(
                    nn.Conv2d(condition_channels, hidden_channels, kernel_size=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True),
                )
            final_offset_mask = (
                self.offset_mask[-1]
                if isinstance(self.offset_mask, nn.Sequential)
                else self.offset_mask
            )
            nn.init.constant_(final_offset_mask.weight, 0.0)
            if final_offset_mask.bias is None:
                raise RuntimeError("offset_mask.bias is required when bias=True.")
            nn.init.constant_(final_offset_mask.bias, 0.0)
        else:
            self.offset_mask = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        deform_conv2d = tv_deform_conv2d
        if not self.use_deformable or deform_conv2d is None:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                padding=self.padding,
                groups=self.align_groups,
            )

        if self.offset_mask is None:
            raise RuntimeError("offset_mask is required when deformable convolution is enabled.")

        offset_mask = self.offset_mask(cond)
        k2 = self.kernel_size * self.kernel_size
        offset = offset_mask[:, : 2 * k2]
        if self.max_offset is not None:
            offset = self.max_offset * torch.tanh(offset)
        mask = torch.sigmoid(offset_mask[:, 2 * k2 :])

        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(self.padding, self.padding),
            dilation=(1, 1),
            mask=mask,
        )


class DualFrequencySkipFusion(nn.Module):
    """轻量同尺度跳连融合模块：保留 base 相加，再对 skip 做条件可变形对齐。"""

    def __init__(
        self,
        channels: int,
        block_type: str | None = None,
        reduction: int = 4,
        dfpb_kwargs: dict | None = None,
        use_deformable: bool = True,
        align_kernel_size: int = 3,
        align_groups: int | str = 1,
        offset_reduction: int = 1,
        max_offset: float | None = None,
        **legacy_kwargs,
    ) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if align_kernel_size % 2 == 0:
            raise ValueError("align_kernel_size must be odd.")

        _ = block_type
        _ = dfpb_kwargs
        _ = legacy_kwargs

        self.channels = channels
        self.reduction = reduction
        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.dec_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.align_skip = ConditionedDeformableConv2d(
            channels=channels,
            condition_channels=channels * 2,
            kernel_size=align_kernel_size,
            padding=align_kernel_size // 2,
            use_deformable=use_deformable,
            align_groups=align_groups,
            offset_reduction=offset_reduction,
            max_offset=max_offset,
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        if x_dec.shape != x_skip.shape:
            raise ValueError(
                "DualFrequencySkipFusion expects same-scale, same-channel features; "
                f"got x_dec={tuple(x_dec.shape)} and x_skip={tuple(x_skip.shape)}."
            )

        base = x_dec + x_skip
        dec = self.dec_proj(self.norm_dec(x_dec))
        skip = self.skip_proj(self.norm_skip(x_skip))
        cond = torch.cat([dec, skip], dim=1)
        aligned_skip = self.align_skip(skip, cond)
        out = base + self.gamma * self.out_proj(aligned_skip)

        self._last_aux = {
            "gamma_abs_mean": self.gamma.abs().mean().detach(),
            "aligned_skip_abs_mean": aligned_skip.abs().mean().detach(),
            "use_deformable": torch.tensor(
                float(self.align_skip.use_deformable),
                device=x_dec.device,
                dtype=x_dec.dtype,
            ),
        }
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


class ReliableDeformSkipFusion(nn.Module):
    """Reliability-calibrated deformable skip fusion."""

    def __init__(
        self,
        channels: int,
        block_type: str | None = None,
        reduction: int = 4,
        dfpb_kwargs: dict | None = None,
        use_deformable: bool = True,
        align_kernel_size: int = 3,
        align_groups: int | str = 1,
        offset_reduction: int = 1,
        max_offset: float | None = None,
        confidence_reduction: int = 4,
        confidence_init: float = 2.0,
        suppress_scale: float = 0.2,
        **legacy_kwargs,
    ) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if confidence_reduction <= 0:
            raise ValueError("confidence_reduction must be positive.")
        if align_kernel_size % 2 == 0:
            raise ValueError("align_kernel_size must be odd.")

        _ = block_type
        _ = dfpb_kwargs
        _ = legacy_kwargs

        self.channels = channels
        self.reduction = reduction
        self.suppress_scale = float(suppress_scale)
        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.dec_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.align_skip = ConditionedDeformableConv2d(
            channels=channels,
            condition_channels=channels * 2,
            kernel_size=align_kernel_size,
            padding=align_kernel_size // 2,
            use_deformable=use_deformable,
            align_groups=align_groups,
            offset_reduction=offset_reduction,
            max_offset=max_offset,
        )

        hidden_channels = max(channels // confidence_reduction, 8)
        self.conf_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
        )
        nn.init.constant_(self.conf_head[-1].weight, 0.0)
        nn.init.constant_(self.conf_head[-1].bias, confidence_init)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        if x_dec.shape != x_skip.shape:
            raise ValueError(
                "ReliableDeformSkipFusion expects same-scale, same-channel features; "
                f"got x_dec={tuple(x_dec.shape)} and x_skip={tuple(x_skip.shape)}."
            )

        base = x_dec + x_skip
        dec = self.dec_proj(self.norm_dec(x_dec))
        skip = self.skip_proj(self.norm_skip(x_skip))
        cond = torch.cat([dec, skip], dim=1)

        consistency = (dec - skip).abs().mean(dim=1, keepdim=True)
        confidence = torch.sigmoid(self.conf_head(torch.cat([dec, skip, consistency], dim=1)))
        aligned_skip = self.align_skip(skip, cond)
        detail = self.out_proj(aligned_skip)
        reliable_detail = confidence * detail - self.suppress_scale * (1.0 - confidence) * x_skip
        out = base + self.gamma * reliable_detail

        self._last_aux = {
            "gamma_abs_mean": self.gamma.abs().mean().detach(),
            "confidence_mean": confidence.mean().detach(),
            "confidence_std": confidence.std().detach(),
            "aligned_skip_abs_mean": aligned_skip.abs().mean().detach(),
            "use_deformable": torch.tensor(
                float(self.align_skip.use_deformable),
                device=x_dec.device,
                dtype=x_dec.dtype,
            ),
        }
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


DualFrequencySkipFusionDeform = DualFrequencySkipFusion
DualFrequencySkipFusionLite = DualFrequencySkipFusion

__all__ = [
    "LayerNorm2d",
    "ConditionedDeformableConv2d",
    "DualFrequencySkipFusion",
    "ReliableDeformSkipFusion",
    "DualFrequencySkipFusionDeform",
    "DualFrequencySkipFusionLite",
]
