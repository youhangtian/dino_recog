import os 
import math
import torch 
import torch.nn as nn 


@torch.no_grad()
def save_model(student, teacher, img_size, path, name, save_onnx=False):
    save_dict = {'student': student.module.state_dict(), 'teacher': teacher.state_dict()}
    save_path = os.path.join(path, f'{name}.pth')
    torch.save(save_dict, save_path)

    if save_onnx:
        img =  torch.randn(1, 3, img_size[0], img_size[1]).cuda()
        save_path = os.path.join(path, f'{name}.onnx')
        torch.onnx.export(
            teacher.backbone,
            img, 
            save_path,
            input_names=['input'],
            output_names=['features'],
            dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}},
            opset_version=15
        )


def get_model(student, teacher , cfg, logger=None):
    student = MultiCropWrapper(
        student,
        DINOHead(cfg.num_features, cfg.out_dim, norm_last_layer=True)
    )
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(cfg.num_features, cfg.out_dim)
    )

    if cfg.ckpt:
        if logger: logger.info(f'load checkpoint from {cfg.ckpt} ------')
        checkpoint = torch.load(cfg.ckpt, map_location='cpu')
        student.load_state_dict(checkpoint['student'], strict=False)
        teacher.load_state_dict(checkpoint['teacher'], strict=False)

    return student, teacher

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


class DINOHead(nn.Module):
    def __init__(self,
                 in_dim, 
                 out_dim,
                 norm_last_layer=True,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False 

        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):    
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x) 
        return x 
        

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone 
        self.head = head 

    def forward(self, x, return_attention=False):
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output, attn = 0, torch.empty(0).to(x[0].device), torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            if return_attention:
                _out = self.backbone(torch.cat(x[start_idx:end_idx]), return_attention=return_attention)
                output = torch.cat((output, _out[0]))
                attn = torch.cat((attn, _out[1]))
            else:
                _out = self.backbone(torch.cat(x[start_idx:end_idx]))
                output = torch.cat((output, _out))
            start_idx = end_idx 

        if return_attention:
            return self.head(output), attn 
        else:
            return self.head(output)
