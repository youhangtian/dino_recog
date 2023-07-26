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


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dims, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dims, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = self.proj(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, out_size 


class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding='corner'):
        super(AdaptivePadding, self).__init__()
        self.padding = padding 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.dilation = dilation 

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape 
        kernel_h, kernel_w = self.kernel_size 
        stride_h, stride_w = self.stride 
        dilation_h, dilation_w = self.dilation
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)

        return pad_h, pad_w 
    
    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = nn.functional.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = nn.functional.pad(x, [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2, 
                    pad_h - pad_h // 2
                ])
            else:
                raise ValueError(f'do not support padding {self.padding}')
            
        return x 
    

class PatchMerging(nn.Module):
    def __init__(self,
                 in_chans, 
                 out_chans,
                 kernel_size=(2, 2),
                 stride=(2, 2),
                 dilation=(1, 1),
                 padding='corner'):
        super().__init__()
        self.in_chans = in_chans 
        self.out_chans = out_chans 

        self.adap_padding = AdaptivePadding(
            kernel_size=kernel_size, 
            stride=stride, 
            dilation=dilation,
            padding=padding
        )
        padding = (0, 0)
        
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride
        )

        sample_dim = kernel_size[0] * kernel_size[1] * in_chans 
        self.norm = nn.LayerNorm(sample_dim)
        self.reduction = nn.Linear(sample_dim, out_chans, bias=False)

    def forward(self, x, input_size):
        B, L, C = x.shape
        H, W = input_size 
        assert L == H * W, f'wrong shape {L} {H} {W}'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        x = self.adap_padding(x)
        H, W = x.shape[-2:]

        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * 
                 (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1
        out_size = (out_h, out_w)

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.reduction(x)
        return x, out_size 


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob 

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x 
        
        keep_prob = 1 - self.drop_prob 

        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor.floor()

        return output 


class WindowMSA(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim 
        self.window_size = window_size 
        self.num_heads = num_heads 
        head_embed_dim = embed_dim // num_heads 
        self.scale = qk_scale or head_embed_dim ** -0.5 

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
            num_heads
            )
        )

        Wh, Ww = self.window_size 
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T 
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 
    
    @staticmethod 
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0):
        super().__init__()

        self.window_size = window_size 
        self.shift_size = shift_size 
        assert 0 <= self.shift_size < self.window_size 

        self.w_msa = WindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=(window_size, window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate
        )

    def forward(self, query, hw_shape):
        B, L, C = query.shape 
        H, W = hw_shape 
        assert L == H * W, f'wrong shape {L}, {H}, {W}'
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size 
        pad_b = (self.window_size - H % self.window_size) % self.window_size 
        query = nn.functional.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1,2)
            )

            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            cnt = 0 
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt 
                    cnt += 1 

            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask==0, float(0.0))
        else:
            shifted_query = query 
            attn_mask = None 

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)

        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, 
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x 

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous() 

        x = x.view(B, H * W, C)
        return x 
    
    def window_reverse(self, windows, H, W):
        window_size = self.window_size 
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x 
    
    def window_partition(self, x):
        B, H, W, C = x.shape 
        window_size = self.window_size 
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows 
    

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 


class SwinBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 hidden_chans,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):
        super(SwinBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ShiftWindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=hidden_chans,
            out_features=embed_dim,
            drop=drop_rate
        )

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 


class SwinLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads, 
                 hidden_chans,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_chans=hidden_chans,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            self.blocks.append(block)

        self.downsample = downsample 

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x, hw_shape = self.downsample(x, hw_shape)

        return x, hw_shape 


class SwinTransformer(nn.Module):
    def __init__(self,
                 fp16=False, 
                 input_size=(224, 224),
                 in_chans=3,
                 num_features=512,
                 patch_size=4, 
                 embed_dim=96,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 ckpt=None):
        super(SwinTransformer, self).__init__()
        self.fp16 = fp16 

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size=patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            downsample = PatchMerging(embed_dim, embed_dim*2) if i < len(depths) - 1 else None
            layer = SwinLayer(
                embed_dim=embed_dim,
                num_heads=num_heads[i],
                hidden_chans=mlp_ratio * embed_dim,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i+1])],
                downsample=downsample
            )

            self.layers.append(layer)
            if downsample: embed_dim = downsample.out_chans 

        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.features = nn.Linear(in_features=embed_dim, out_features=num_features, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x, hw_shape = self.patch_embed(x)
            x = self.pos_drop(x) 

            for layer in self.layers:
                x, hw_shape = layer(x, hw_shape)

            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.flatten(1)
            features = self.features(x)

        if self.fp16: features = features.float()
        features = nn.functional.normalize(features, dim=-1, p=2)

        return features 

