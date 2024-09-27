# References:
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py

import torch
from torch import nn
from torch.nn import functional as F
import contextlib


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, time_channels, max_len=4000):
        super().__init__()

        self.d_model = time_channels // 4

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(self.d_model // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.d_model))
        pe_enc_mat = torch.zeros(size=(max_len, self.d_model))
        pe_enc_mat[:, 0:: 2] = torch.sin(angle)
        pe_enc_mat[:, 1:: 2] = torch.cos(angle)

        self.register_buffer("pos_enc_mat", pe_enc_mat)

    def forward(self, diffusion_step):
        return self.pos_enc_mat[diffusion_step]


class ResConvSelfAttn(nn.Module):
    def __init__(self, channels, n_groups=32):
        super().__init__()

        self.gn = nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1, 1, 0)
        self.out_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.scale = channels ** (-0.5)

    def forward(self, x):
        b, c, h, w = x.shape
        skip = x

        x = self.gn(x)
        x = self.qkv_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=1)
        attn_score = torch.einsum(
            "bci,bcj->bij", q.view((b, c, -1)), k.view((b, c, -1)),
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=2)        
        x = torch.einsum("bij,bcj->bci", attn_weight, v.view((b, c, -1)))
        x = x.view(b, c, h, w)
        x = self.out_proj(x)
        return x + skip


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_channels, attn=False, n_groups=32, drop_prob=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn = attn

        self.layers1 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_channels, out_channels),
        )
        self.layers2 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv = nn.Identity()

        if attn:
            self.attn_block = ResConvSelfAttn(out_channels)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        skip = x
        x = self.layers1(x)
        x = x + self.time_proj(t)[:, :, None, None]
        x = self.layers2(x)
        x = x + self.conv(skip)
        return self.attn_block(x)


class Downsample(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels, channels, 3, 2, 1)


class Classifier(nn.Module):
    def __init__(
        self,
        n_classes,
        channels=128,
        channel_mults=[1, 2, 2, 2],
        attns=[False, True, False, False],
        n_res_blocks=2,
    ):
        super().__init__()

        assert all([i < len(channel_mults) for i in attns]), "attns index out of bound"

        time_channels = channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(max_len=1000, time_channels=time_channels),
            nn.Linear(channels, time_channels),
            Swish(),
            nn.Linear(time_channels, time_channels),
        )
        self.label_emb = nn.Embedding(n_classes, time_channels)

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        cur_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            for _ in range(n_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        attn=attns[i]
                    )
                )
                cur_channels = out_channels
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(cur_channels))

        self.mid_blocks = nn.ModuleList([
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=True,
            ),
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=False,
            ),
        ])

        self.fin_block = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            Swish(),
            nn.Conv2d(cur_channels, n_classes, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step, label):
        x = self.init_conv(noisy_image)
        t = self.time_embed(diffusion_step)
        y = self.label_emb(label)

        for layer in self.down_blocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t + y)

        for layer in self.mid_blocks:
            x = layer(x, t + y)

        x = self.fin_block(x)
        return x.mean(dim=(2, 3))

    def get_loss(self, noisy_image, diffusion_step, label):
        device = next(self.parameters()).device
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
        ) if device.type == "cuda" else contextlib.nullcontext():
            pred = self(
                noisy_image=noisy_image,
                diffusion_step=diffusion_step,
                label=label,
            )
            return F.cross_entropy(pred, label, reduction="mean")

    def get_acc(self, noisy_image, diffusion_step, label, k=1):
        pred = self(
            noisy_image=noisy_image,
            diffusion_step=diffusion_step,
            label=label,
        )
        _, top_k = torch.topk(pred, k=k, dim=1)
        corr = torch.eq(top_k, label.unsqueeze(1).repeat(1, k))
        acc = corr.sum(dim=1).float().mean().item()
        return acc


if __name__ == "__main__":
    classifier = Classifier(n_classes=10)

    noisy_image = torch.randn(4, 3, 32, 32)
    diffusion_step = torch.randint(0, 1000, size=(4,))
    label = torch.randint(0, 10, size=(4,))

    noisy_image.requires_grad = True
    out = classifier(noisy_image, diffusion_step, label)
    # noisy_image.requires_grad, diffusion_step.requires_grad, label.requires_grad, out.requires_grad

    grad1 = torch.autograd.grad(out.sum(), noisy_image)[0]
    grad1.sum()
    # grad2 = torch.autograd.grad(out.mean(), noisy_image)[0]
    # grad2.sum()
