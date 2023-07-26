import torch 
import torch.nn as nn 
from functools import partial 

from .resnet import ResNet 
from .vision_transformer import VisionTransformer 
from .swin_transformer import SwinTransformer 

def get_backbone(cfg, logger=None):
    if cfg.network == 'r18':
        model = ResNet(
            [2, 2, 2, 2],
            cfg.input_size, 
            num_features=cfg.num_features,
            fp16=cfg.fp16
        )
    elif cfg.network == 'r34':
        model = ResNet(
            [3, 4, 6, 3],
            cfg.input_size, 
            num_features=cfg.num_features,
            fp16=cfg.fp16
        )
    elif cfg.network == 'r50':
        model = ResNet(
            [3, 4, 14, 3],
            cfg.input_size, 
            num_features=cfg.num_features,
            fp16=cfg.fp16
        )
    elif cfg.network == 'r100':
        model = ResNet(
            [3, 13, 30, 3],
            cfg.input_size, 
            num_features=cfg.num_features,
            fp16=cfg.fp16
        )
    elif cfg.network == 'r200':
        model = ResNet(
            [6, 26, 60, 6],
            cfg.input_size, 
            num_features=cfg.num_features,
            fp16=cfg.fp16
        )

    elif cfg.network == 'vit_s':
        model = VisionTransformer(
            fp16=cfg.fp16,
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            num_features=cfg.num_features,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif cfg.network == 'vit_b':
        model = VisionTransformer(
            fp16=cfg.fp16,
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            num_features=cfg.num_features,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    elif cfg.network == 'swin_f':
        model = SwinTransformer(
            fp16=cfg.fp16,
            input_size=cfg.input_size,
            num_features=cfg.num_features,
            embed_dim=64,
            num_heads=(2, 4, 8, 16),
            patch_size=2,
        )
    elif cfg.network == 'swin_s':
        model = SwinTransformer(
            fp16=cfg.fp16,
            input_size=cfg.input_size,
            num_features=cfg.num_features,
            embed_dim=64,
            num_heads=(2, 4, 8, 16)
        )

    else:
        raise ValueError(f'do not support backbone {cfg.network}')

    if cfg.ckpt:
        if logger: logger.info(f'load checkpoint from {cfg.ckpt} ------')
        checkpoint = torch.load(cfg.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    return model
