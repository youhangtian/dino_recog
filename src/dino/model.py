import os 
import math
import torch 
import torch.nn as nn 


@torch.no_grad()
def save_model(student, teacher, img_size, path, name, save_onnx=False):
    save_dict = {'student': student.module.state_dict(), 'teacher': teacher.state_dict()}
    save_path = os.path.join(path, f'{name}.pth')
    torch.save(save_dict, save_path)

    backbone_save_dict = teacher.backbone.state_dict()
    backbone_save_path = os.path.join(path, f'{name}_backbone.pth')
    torch.save(backbone_save_dict, backbone_save_path)

    if save_onnx:
        img =  torch.randn(1, 3, img_size[0], img_size[1]).cuda()
        save_path = os.path.join(path, f'{name}.onnx')
        torch.onnx.export(
            teacher.backbone,
            img, 
            save_path,
            input_names=['input'],
            output_names=['features', 'features_norm'],
            dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}, 'features_norm': {0: 'batch_size'}},
            opset_version=15
        )


def get_model(student_backbone, teacher_backbone , cfg, logger=None):
    student = DinoWrapper(
        student_backbone,
        Head(cfg.num_features, cfg.out_dim, norm_last_layer=True),
        Head(cfg.num_features, cfg.patch_out_dim, norm_last_layer=True)
    )
    teacher = DinoWrapper(
        teacher_backbone,
        Head(cfg.num_features, cfg.out_dim),
        Head(cfg.num_features, cfg.patch_out_dim)
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


class Head(nn.Module):
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        output = self.last_layer(x) 
        return output
        

class DinoWrapper(nn.Module):
    def __init__(self, backbone, head1, head2):
        super(DinoWrapper, self).__init__()
        self.backbone = backbone 
        self.head1 = head1
        self.head2 = head2

    def forward(self, x, masks=None, return_attention=False):
        x = torch.cat(x)
        masks = torch.cat(masks) if masks is not None else None

        features, features_norm, features_all, attn = self.backbone(x, masks, return_all=True)
        output1 = self.head1(features)
        output2 = self.head2(features_all[:, 1:])

        if return_attention:
            return output1, output2, attn 
        else:
            return output1, output2
