from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn


def _exists(value):
    return value is not None


def _default(value, fallback):
    return value if _exists(value) else fallback


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def Upsample(dim: int, dim_out: int | None = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: int | None = None):
    return nn.Conv2d(dim, _default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = reduce(self.weight, "o ... -> o 1 1 1", "mean")
        var = reduce(self.weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        weight = (self.weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        if dim % 2:
            raise ValueError("learned_sinusoidal_dim must be even")
        self.weights = nn.Parameter(torch.randn(dim // 2), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        return torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, groups: int = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if _exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim=None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if _exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if _exists(self.mlp) and _exists(time_emb):
            scale_shift = rearrange(self.mlp(time_emb), "b c -> b c 1 1").chunk(2, dim=1)
        h = self.block2(self.block1(x, scale_shift=scale_shift))
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = map(
            lambda t: rearrange(t, "b (head c) x y -> b head c (x y)", head=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        v = v / (h * w)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b head c (x y) -> b (head c) x y", head=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = map(
            lambda t: rearrange(t, "b (head c) x y -> b head c (x y)", head=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        sim = einsum("b h d i, b h d j -> b h i j", q * self.scale, k)
        out = einsum("b h i j, b h d j -> b h i d", sim.softmax(dim=-1), v)
        out = rearrange(out, "b head (x y) d -> b (head d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels: int = 3,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        condition: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * int(condition)
        init_dim = _default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        if learned_sinusoidal_cond or random_fourier_features:
            pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            nn.Conv2d(dim_in, dim_out, 3, padding=1)
                            if is_last
                            else Downsample(dim_in, dim_out)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            nn.Conv2d(dim_out, dim_in, 3, padding=1)
                            if is_last
                            else Upsample(dim_out, dim_in)
                        ),
                    ]
                )
            )

        self.out_dim = _default(out_dim, channels * (2 if learned_variance else 1))
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        height, width = x.shape[-2:]
        multiple = 2**self.depth
        x = F.pad(
            x,
            (0, (multiple - width % multiple) % multiple, 0, (multiple - height % multiple) % multiple),
            mode="reflect",
        )
        x = self.init_conv(x)
        residual = x.clone()
        time = self.time_mlp(time)
        skips = []
        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time)
            skips.append(x)
            x = attention(block2(x, time))
            skips.append(x)
            x = downsample(x)
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, time)), time)
        for block1, block2, attention, upsample in self.ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), time)
            x = attention(block2(torch.cat((x, skips.pop()), dim=1), time))
            x = upsample(x)
        x = self.final_res_block(torch.cat((x, residual), dim=1), time)
        return self.final_conv(x)[..., :height, :width].contiguous()


class UnetRes(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels: int = 4,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        num_unet: int = 1,
        condition: bool = False,
        objective: str = "pred_res_noise",
        test_res_or_noise: str = "res_noise",
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        self.out_dim = _default(out_dim, channels * (2 if learned_variance else 1))
        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        self.unet0 = Unet(
            dim,
            init_dim=init_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            condition=condition,
        )

    def forward(self, x, time):
        return self.unet0(x, time)


class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)

    def forward(self, text_features, image_feature):
        query = self.query_linear(image_feature).repeat(1, 77, 1)
        key = self.key_linear(text_features)
        value = self.value_linear(text_features)
        weights = F.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)
        return torch.bmm(weights, value)


class SimplifiedTransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_projection = (
            nn.Linear(embedding_dim, vocab_size - embedding_dim)
            if vocab_size > embedding_dim
            else None
        )

    def forward(self, target, memory):
        output = self.decoder(target, memory)
        if self.output_projection is not None:
            output = torch.cat([output, self.output_projection(output)], dim=2)
        return output


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.output_layer(self.activation(self.hidden_layer(x)))


class LightweightImageToTextTransformerModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        image_feature_dim: int = 768,
        num_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 4,
        vocab_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_att = CrossAttention(image_feature_dim)
        self.decoder = SimplifiedTransformerDecoder(
            embedding_dim, num_heads, ffn_dim, num_layers, vocab_size, dropout
        )
        self.clip_proj = nn.ModuleList(
            [SimpleMLP(image_feature_dim, image_feature_dim, image_feature_dim) for _ in range(2)]
        )

    def forward(self, tgt_embeddings, image_features, t=None, **_):
        condition = image_features
        for projection in self.clip_proj:
            condition = projection(condition)
        condition_token = condition.unsqueeze(1)
        target = self.cross_att(tgt_embeddings, condition_token).permute(1, 0, 2)
        output = self.decoder(target, condition.unsqueeze(0)).permute(1, 0, 2)
        return output + condition_token


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (var + eps).sqrt()
        ctx.save_for_backward(normalized, var, weight)
        return weight.view(1, channels, 1, 1) * normalized + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        normalized, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_norm = (grad * normalized).mean(dim=1, keepdim=True)
        grad_x = (grad - normalized * mean_grad_norm - mean_grad) / torch.sqrt(var + ctx.eps)
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    def __init__(self, channels: int, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channels = channels * DW_Expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, 3, padding=1, groups=dw_channels
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.sg = SimpleGate()
        ffn_channels = channels * FFN_Expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.dropout1(self.conv3(x * self.sca(x)))
        y = inp + x * self.beta
        x = self.dropout2(self.conv5(self.sg(self.conv4(self.norm2(y)))))
        return y + x * self.gamma


class NAFNet_Combine(nn.Module):
    def __init__(
        self,
        img_channel: int = 6,
        width: int = 64,
        middle_blk_num: int = 12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, 2))
            channels *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])
        for count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        height, width = inp.shape[-2:]
        inp = F.pad(
            inp,
            (
                0,
                (self.padder_size - width % self.padder_size) % self.padder_size,
                0,
                (self.padder_size - height % self.padder_size) % self.padder_size,
            ),
        )
        x = self.intro(inp)
        skips = []
        for encoder, downsample in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)
        x = self.middle_blks(x)
        for decoder, upsample, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(upsample(x) + skip)
        x = self.ending(x) + inp[:, :3]
        return x[..., :height, :width]
