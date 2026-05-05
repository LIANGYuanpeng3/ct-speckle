# model_unet_small_res.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Conv -> Norm -> SiLU
# 用 GroupNorm 替代 BatchNorm：小 batch 更稳
# -----------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, groups_gn=8, groups_conv=1):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=groups_conv, bias=False)

        # GroupNorm：group 数不能超过通道数，且需整除
        g = min(groups_gn, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, out_ch, eps=1e-5)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# -----------------------------
# 轻量 SE（channel attention）
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(ch // r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


# -----------------------------
# ResBlock（可选 dilation）
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, use_se=True, d_rate=1, groups_gn=8):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch, k=3, d=d_rate, groups_gn=groups_gn)
        self.conv2 = ConvGNAct(ch, ch, k=3, d=1, groups_gn=groups_gn)
        self.se = SEBlock(ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return x + out


# -----------------------------
# Down: stride-2 conv
# -----------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, groups_gn=8):
        super().__init__()
        self.block = ConvGNAct(in_ch, out_ch, k=3, s=2, groups_gn=groups_gn)

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Up: interpolate + 1x1 reduce + concat skip
# -----------------------------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, groups_gn=8):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        g = min(groups_gn, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, out_ch, eps=1e-5)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.act(self.norm(self.reduce(x)))
        return torch.cat([x, skip], dim=1)


# -----------------------------
# UNetSmallRes
# - 3-level UNet (H/1, H/2, H/4) + bottleneck
# - residual output (y += input) by default
# -----------------------------
class UNetSmallRes(nn.Module):
    """
    in_ch=1: 你的 LF 输入
    out_ch=1: 预测 SF（或 GT）
    base: 宽度（32/48/64）
    use_se: 是否启用 SE
    use_residual: 是否做 residual learning (y = y + x[:, :1])
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, use_se=True, use_residual=True, groups_gn=8):
        super().__init__()
        self.use_residual = use_residual

        # Encoder
        self.enc1 = nn.Sequential(
            ConvGNAct(in_ch, base, k=3, groups_gn=groups_gn),
            ResBlock(base, use_se=use_se, d_rate=1, groups_gn=groups_gn),
        )
        self.down1 = Down(base, base * 2, groups_gn=groups_gn)

        self.enc2 = nn.Sequential(
            ResBlock(base * 2, use_se=use_se, d_rate=2, groups_gn=groups_gn),  # dilation=2 扩感受野
            ResBlock(base * 2, use_se=use_se, d_rate=1, groups_gn=groups_gn),
        )
        self.down2 = Down(base * 2, base * 4, groups_gn=groups_gn)

        # Bottleneck
        self.neck = nn.Sequential(
            ResBlock(base * 4, use_se=use_se, d_rate=2, groups_gn=groups_gn),
            ResBlock(base * 4, use_se=use_se, d_rate=1, groups_gn=groups_gn),
        )

        # Decoder
        self.up2 = Up(base * 4, base * 2, groups_gn=groups_gn)  # -> cat with enc2 => 4b
        self.dec2 = nn.Sequential(
            ConvGNAct(base * 4, base * 2, k=3, groups_gn=groups_gn),
            ResBlock(base * 2, use_se=use_se, d_rate=1, groups_gn=groups_gn),
        )

        self.up1 = Up(base * 2, base, groups_gn=groups_gn)      # -> cat with enc1 => 2b
        self.dec1 = nn.Sequential(
            ConvGNAct(base * 2, base, k=3, groups_gn=groups_gn),
            ResBlock(base, use_se=use_se, d_rate=1, groups_gn=groups_gn),
        )

        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # [B, b, H, W]
        d1 = self.down1(e1)     # [B, 2b, H/2, W/2]
        e2 = self.enc2(d1)      # [B, 2b, H/2, W/2]
        d2 = self.down2(e2)     # [B, 4b, H/4, W/4]

        # Neck
        z = self.neck(d2)       # [B, 4b, H/4, W/4]

        # Decoder
        u2 = self.up2(z, e2)    # cat -> [B, 4b, H/2, W/2]
        u2 = self.dec2(u2)      # [B, 2b, H/2, W/2]

        u1 = self.up1(u2, e1)   # cat -> [B, 2b, H, W]
        u1 = self.dec1(u1)      # [B, b, H, W]

        y = self.out(u1)        # [B, out_ch, H, W]

        # residual learning: y = y + input
        if self.use_residual and x.shape[1] >= 1 and y.shape[1] == 1:
            y = y + x[:, :1, ...]
        return y
