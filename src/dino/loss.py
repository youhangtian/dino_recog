import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np 

class DINOLoss(nn.Module):
    def __init__(self, 
                 out_dim, 
                 patch_out_dim, 
                 global_crops_number,
                 local_crops_number,
                 cfg):
        super().__init__()
        self.student_temp = cfg.student_temp 
        self.center_momentum = cfg.center_momentum
        self.global_crops_number = global_crops_number 
        self.local_crops_number = local_crops_number 
        self.ncrops = global_crops_number + local_crops_number 

        self.register_buffer('center', torch.zeros(1, out_dim))
        self.register_buffer('center2', torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = cfg.lambda1 
        self.lambda2 = cfg.lambda2 

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(cfg.warmup_teacher_temp,
                        cfg.teacher_temp,
                        cfg.warmup_teacher_temp_epochs),
            np.ones(cfg.epochs - cfg.warmup_teacher_temp_epochs) * cfg.teacher_temp
        ))

    def forward(self, student_output_global, student_output_local, student_masks, teacher_output, epoch):
        student_global_cls, student_patch = student_output_global[0], student_output_global[1]
        student_local_cls = student_output_local[0]
        student_cls = torch.cat([student_global_cls, student_local_cls])

        student_cls = student_cls / self.student_temp 
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp 
        student_patch_c = student_patch.chunk(self.global_crops_number)

        teacher_cls, teacher_patch = teacher_output[0], teacher_output[1]
        temp = self.teacher_temp_schedule[epoch]
        
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.global_crops_number)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.global_crops_number) 

        total_loss1, total_loss2 = 0, 0
        n_loss_terms1, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_masks[v]
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2 
        total_loss = total_loss1 + total_loss2 
        self.update_center(teacher_cls, teacher_patch)
        return total_loss1, total_loss2, total_loss 
    
    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum + patch_center * (1 - self.center_momentum)
        