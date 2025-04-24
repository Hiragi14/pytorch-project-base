# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from base.base_model import BaseModel
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

# class DynamicRangeFreqencySelectionBlock(nn.Module):
#     def __init__(self, subpatch_num):
#         super().__init__()
#         self.softmax = nn.Softmax(dim=0)
#         self.weights_for_select_freq = nn.Parameter(torch.zeros(subpatch_num), requires_grad=False)

#     def forward(self, x):
#         # x: tuple of sub-patches
#         self.weights_for_select_freq = [self.weights_for_select_freq[m].max() - self.weights_for_select_freq[m].min() for m in self.weights_for_select_freq]
#         weight_for_select_freq = self.softmax(self.weights_for_select_freq)
#         x = (weight_for_select_freq[:,None, None] * x).sum(dim=1)
#         return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, subpatch_num, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # self.softmax = nn.Softmax(dim=0)
        # self.weights_for_select_freq = nn.Parameter(torch.randn(depth, subpatch_num))
        self.sigmoid = nn.Sigmoid()
        self.weights = nn.Parameter(torch.zeros(depth))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        # input: (B, P, D)
        # B: batch size, P: number of patches, D: dimension
        insert = x
        count = 0
        for attn, ff in self.layers:
            weight = self.sigmoid(self.weights[count])
            x = insert if count == 0 else weight * insert + (1 - weight) * x
            x = attn(x) + x
            x = ff(x) + x
            count += 1

        return self.norm(x)

class DynamicRangeViT(BaseModel):
    def __init__(self, *, image_size, patch_size, subpatch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        subpatch_height, subpatch_width = pair(subpatch_size)
        self.patch_size = patch_size
        self.subpatch_size = subpatch_size
        self.dim = dim
        subpatch_num = int(patch_height/subpatch_height) * int(patch_width/subpatch_width)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * subpatch_height * subpatch_width
        
        self.to_patch_embedding_freq = nn.Sequential(
            Rearrange('b n p1 p2 c -> b n (p1 p2 c)', p1 = subpatch_height, p2 = subpatch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, subpatch_num, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.softmax = nn.Softmax(dim=0)
        self.weights_for_select_freq = nn.Parameter(torch.zeros(subpatch_num), requires_grad=False)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, subpatch_num, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # img: (B, C, H, W) -> x: (B, N, P, D)
        # B: batch size, C: channels, H: height, W: width
        # N: number of frequency, P: number of patches, D: dimension
        patches_per_freq = self.split_image_to_subpatches(img, self.patch_size, self.subpatch_size)
        embedded = [self.to_patch_embedding_freq(patch) for patch in patches_per_freq]
        # embeded[i]: (B, H*W, D)
        x = torch.stack(embedded, dim=1)
        b, _, n, _ = x.shape
        
        # positional embedding
        x += self.pos_embedding[:, :, :n]
        # embedding dropout
        x = self.dropout(x)

        weights = self.set_dynamic_range(patches_per_freq)
        weights = self.softmax(weights)
        x = (weights[:,None, None] * x).sum(dim=1)

        x = self.transformer(x)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def split_image_to_subpatches(self, image, patch_size, subpatch_size):
        """
        パッチ分割された画像テンソル (B, N, H, W, C) を指定されたサブパッチサイズに分割し、
        各サブパッチをタプルの要素として返す。

        Args:
            image (torch.Tensor): 入力画像テンソル (B, N, H, W, C)
            patch_size (Tuple[int, int]): サブパッチサイズ (ph, pw)

        Returns:
            Tuple[torch.Tensor, ...]: 各パッチを要素とするタプル
        """
        patch_height, patch_width = pair(patch_size)
        ph, pw = pair(subpatch_size)

        assert patch_height % ph == 0 and patch_width % pw == 0, "パッチサイズはサブパッチサイズで割り切れる必要があります"

        patched_img = rearrange(image, 'b c (h p1) (w p2) -> b (h w) p1 p2 c', p1 = patch_height, p2 = patch_width)

        patches = []
        for i in range(0, patch_height, ph):
            for j in range(0, patch_width, pw):
                patch = patched_img[:, :, j:j+ph, i:i+pw, :]
                patches.append(patch)

        return tuple(patches)

    def set_dynamic_range(self, patches):
        weights = nn.Parameter(torch.randn(len(patches)), requires_grad=False)
        B, N, _, _, _ = patches[0].shape

        for i in range(len(patches)):
            max, _ = patches[i].reshape(B, N, -1).max(dim=2)
            min, _ = patches[i].reshape(B, N, -1).min(dim=2)
            weights[i] = max.mean() - min.mean()

        return weights