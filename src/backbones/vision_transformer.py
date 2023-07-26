import math
import torch 
import torch.nn as nn 


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2. 
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor 


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob 

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x 
        
        keep_prob = 1 - self.drop_prob 
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output 
    

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) 
        return x, attn
    

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        
    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
    

class PatchEmbed(nn.Module):
    def __init__(self,
                 input_size=(224, 224),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.input_size = input_size 
        self.patch_size = patch_size  
        self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size) 

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x 
    

class VisionTransformer(nn.Module):
    def __init__(self,
                 fp16=False,
                 input_size=(224, 224),
                 patch_size=16,
                 in_chans=3,
                 num_features=512,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.fp16 = fp16
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_features = num_features 
        self.embed_dim = embed_dim 

        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[i],
            norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.features = nn.Linear(embed_dim, num_features, bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):        
        if (self.input_size[0] == h) and (self.input_size[1] == w):
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        H0 = self.input_size[0] // self.patch_size
        W0 = self.input_size[1] // self.patch_size
        h0 = h // self.patch_size + 0.1
        w0 = w // self.patch_size + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, H0, W0, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / H0, w0 / W0),
            mode='bicubic'
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        B, C, H, W = x.shape 
        x = self.patch_embed(x) 

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, H, W)
        return self.pos_drop(x)
    
    def forward(self, x, return_attention=False):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.prepare_tokens(x)
            for blk in self.blocks:
                x, attn = blk(x)
            x = self.norm(x)

            features = self.features(x[:, 0])
            attn = attn[:, :, 0, 1:]

        if self.fp16:
            features = features.float()
            attn = attn.float()

        features = nn.functional.normalize(features, dim=-1, p=2)
        
        if return_attention:
            return features, attn
        else:
            return features 
