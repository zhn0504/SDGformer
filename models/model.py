import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Local Group Convolution Enhancer
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.spatial_gate = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        channel_weight = self.avg_pool(x).view(b, c)
        channel_weight = self.channel_mlp(channel_weight).view(b, c, 1, 1)
        x = x * channel_weight

        spatial_weight = self.spatial_gate(self.spatial_conv(x))
        return x * spatial_weight


class LGCE(nn.Module):
    def __init__(self, channels, num_groups=8, use_pre_conv=True, use_post_conv=True):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f'channels ({channels}) must be divisible by num_groups ({num_groups}).')

        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1) if use_pre_conv else None
        self.num_groups = num_groups
        self.group_attention = nn.ModuleList([CBAM(channels // num_groups) for _ in range(num_groups)])
        self.activation = nn.Sigmoid()
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1) if use_post_conv else None

    def forward(self, x):
        residual = x
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        group_features = torch.split(x, x.size(1) // self.num_groups, dim=1)
        mask_parts = []
        for group_feature, attention_block in zip(group_features, self.group_attention):
            attention_response = self.activation(attention_block(group_feature))
            threshold = attention_response.mean(dim=(1, 2, 3), keepdim=True)
            mask_part = torch.where(attention_response > threshold, torch.ones_like(attention_response), attention_response)
            mask_parts.append(mask_part)

        attention_mask = torch.cat(mask_parts, dim=1)
        x = x * attention_mask

        if self.post_conv is not None:
            x = self.post_conv(x)

        return x + residual


##########################################################################
## Mixed-scale Frequency Feed-forward Network
class MFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.ch_3x3 = hidden_features * 3 // 2
        self.ch_5x5 = hidden_features // 2
        self.ch_gate = hidden_features

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dw3x3 = nn.Conv2d(self.ch_3x3, self.ch_3x3, kernel_size=3, padding=1, groups=self.ch_3x3, bias=bias)
        self.dw5x5 = nn.Conv2d(self.ch_5x5, self.ch_5x5, kernel_size=5, padding=2, groups=self.ch_5x5, bias=bias)

        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.patch_size = 8
        self.W = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, x):

        x1, x2 = self.project_in(x).split([self.ch_3x3, self.ch_5x5], dim=1)
        x_gate, x1_2 = self.dw3x3(x1).split([self.ch_gate, self.ch_5x5], dim=1)
        x2 = self.dw5x5(x2)

        x = self.act(x_gate) * torch.cat([x1_2, x2], dim=1)
        x = self.project_out(x)

        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float()) * self.W
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)

        return x


##########################################################################
## Sparse Gate Self-Attention
class SGSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SGSA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.input_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_proj = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim // 2 * 3,
            dim // 2 * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim // 2 * 3,
            bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn_scale1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_scale2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_scale3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn_scale4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.dynamic_gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.local_enhancer = LGCE(dim // 2, num_groups=8, use_pre_conv=True, use_post_conv=True)

    def forward(self, x):
        b, _, h, w = x.shape
        local_branch, global_branch = self.input_proj(x).chunk(2, dim=1)
        local_branch = self.local_enhancer(local_branch)

        qkv = self.qkv_dwconv(self.qkv_proj(global_branch))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, channels_per_head, _ = q.shape
        dynamic_k = int(channels_per_head * self.dynamic_gate(global_branch).view(b, -1).mean())
        dynamic_k = max(1, min(channels_per_head, dynamic_k))

        attn_weights = (q @ k.transpose(-2, -1)) * self.temperature
        sparse_mask = torch.zeros(
            b,
            self.num_heads,
            channels_per_head,
            channels_per_head,
            device=global_branch.device,
            dtype=torch.bool,
            requires_grad=False,
        )
        topk_index = torch.topk(attn_weights, k=dynamic_k, dim=-1, largest=True)[1]
        sparse_mask.scatter_(-1, topk_index, True)
        attn_weights = attn_weights.masked_fill(~sparse_mask, float('-inf'))

        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output * (
            self.attn_scale1 + self.attn_scale2 + self.attn_scale3 + self.attn_scale4
        )
        attn_output = rearrange(attn_output, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fused = torch.cat([attn_output, local_branch], dim=1)
        return self.project_out(fused)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SGSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=7, padding='same', bias=bias)

    def forward(self, x):
        return self.proj(x)


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## SDGformer
class SDGformer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super(SDGformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            ) for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        enc1 = self.patch_embed(inp_img)
        enc1 = self.encoder_level1(enc1)

        enc2 = self.down1_2(enc1)
        enc2 = self.encoder_level2(enc2)

        enc3 = self.down2_3(enc2)
        enc3 = self.encoder_level3(enc3)

        enc4 = self.down3_4(enc3)
        latent = self.latent(enc4)

        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.reduce_chan_level3(dec3)
        dec3 = self.decoder_level3(dec3)

        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_chan_level2(dec2)
        dec2 = self.decoder_level2(dec2)

        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder_level1(dec1)

        out = self.refinement(dec1)
        out = self.output(out) + inp_img
        
        return [out]


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SDGformer().to(device)
    x = torch.randn(2, 3, 64, 64).to(device)
    y = model(x)
    print(y.shape)

    flops, params = profile(model, inputs=(x,))
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs: {flops / 1e9:.2f} G")

    