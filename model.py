import torch

# from vit_pytorch import ViT
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor
from torch.nn.functional import gelu, layer_norm
from torch.nn.modules import Transformer, TransformerEncoder
from torch.nn.modules.transformer import F

import wandb


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )  # this breaks down the image in s1xs2 patches, and then flat them

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # prepending the cls token
        x += self.positions
        return x


class MLP(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        for units in self.hidden_units:
            self.layers.append(nn.Linear(in_features=units, out_features=units))
            self.layers.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ViTforObjectDetection(nn.Module):
    def __init__(
        self,
        mlp_head_units: list[int],
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        num_heads: int = 4,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.inputs = torch.Tensor()
        self.mlp = MLP(mlp_head_units, dropout_rate=0.3)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.to_patch_embedding = PatchEmbedding(
            patch_size=patch_size, img_size=img_size, emb_size=emb_size
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=num_heads,
                activation=gelu,
                layer_norm_eps=1e-6,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, img)->Tensor:
        x = self.to_patch_embedding(img)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp(x)
        bounding_box = nn.Linear(x.size(-1), out_features=4)(x)
        return bounding_box
