import os 
import sys
import argparse

import math
import numpy as np 
from tqdm import tqdm 

import torch 
from torch import nn 
from torch import distributed 
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from src.utils import get_config_from_yaml, get_logger, AverageMeter
from src.backbones import get_backbone
from src.megaface_test import get_mega_dataloader, get_acc

from src.dino.model import get_model, save_model 
from src.dino.data import get_dataloader 
from src.dino.utils import cosine_scheduler, get_params_groups 
from src.dino.loss import DINOLoss

def get_config():
    parser = argparse.ArgumentParser(description='training argument')
    parser.add_argument('-c', '--config_file', default='cfg_dino.yaml', type=str, help='config file')
    parser.add_argument('opts', help='modify config options from the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_config_from_yaml(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg 

def main(cfg):
    try:
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        RANK = int(os.environ['LOCAL_RANK'])
        distributed.init_process_group('nccl')
    except KeyError:
        print('------ key error ------')
        WORLD_SIZE = 1
        RANK = 0
        distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:12584',
            rank=RANK,
            world_size=WORLD_SIZE,
        )

    torch.cuda.set_device(RANK)

    os.makedirs(cfg.output, exist_ok=True)

    logger = get_logger(cfg.output, f'log.txt') if RANK == 0 else None 
    if logger: logger.info(f'config: ------\n{cfg} ------')

    if logger: logger.info(f'get dataloader ------\n{cfg.data} ------')
    data_loader = get_dataloader(cfg.data)

    if logger: logger.info(f'get model ------\n{cfg.model.network} ------')
    student = get_backbone(cfg.model, logger=logger)
    teacher = get_backbone(cfg.model, logger=logger)
    student, teacher = get_model(student, teacher, cfg.model, logger=logger)
    student, teacher = student.cuda(), teacher.cuda()

    student = nn.parallel.DistributedDataParallel(student, device_ids=[RANK])
    teacher.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False 

    if logger: logger.info(f'get dino loss ------')
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        cfg.data.local_crops_number + 2,
        cfg.train
    ).cuda()

    if logger: logger.info(f'get optimzer {cfg.train.optimizer} ------')
    params_groups = get_params_groups(student)
    if cfg.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_groups)
    elif cfg.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    else:
        raise ValueError(f'do not support optimiizer {cfg.train.optimizer}')
    
    lr_scheduler = cosine_scheduler(
        cfg.train.lr * (cfg.data.batch_size * WORLD_SIZE) / 256.,
        cfg.train.lr_end,
        cfg.train.epochs,
        len(data_loader),
        warmup_epochs=cfg.train.warmup_epochs
    )

    wd_scheduler = cosine_scheduler(
        cfg.train.weight_decay,
        cfg.train.weight_decay_end,
        cfg.train.epochs,
        len(data_loader)
    )

    momentum_scheduler = cosine_scheduler(
        cfg.train.momentum_teacher,
        1.0,
        cfg.train.epochs,
        len(data_loader)
    )

    face_dataloaders = []
    face_folders = cfg.data.megaface_face_folders.split(',')
    if logger: logger.info(f'get megaface dataloaders {face_folders} ------')

    for i in range(len(face_folders)):
        face_dataloader = get_mega_dataloader(cfg.data.megaface_data_root, face_folders[i], cfg.data.batch_size, cfg.data.resize)
        face_dataloaders.append(face_dataloader)

    best_acc = {name: 0.0 for name in face_folders[:-1]}

    fp16_scaler = torch.cuda.amp.GradScaler() if cfg.model.fp16 else None 

    loss_am = AverageMeter() if RANK == 0 else None 
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'tensorboard')) if RANK == 0 else None 

    steps = -1
    for epoch in range(cfg.train.epochs):
        if logger: logger.info(f'epoch {epoch} begin ------')

        if (epoch % cfg.train.save_epoch == 0) and (RANK == 0):
            save_model(student, teacher, cfg.model.input_size, cfg.output, f'epoch{epoch}', save_onnx=False)

        data_loader.sampler.set_epoch(epoch)

        for (imgs, labels) in tqdm(data_loader, f'[epoch{epoch}][rank{RANK}]'):
            steps += 1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_scheduler[steps]
                if i == 0: param_group['weight_decay'] = wd_scheduler[steps]

            images = [img.cuda(non_blocking=True) for img in imgs]

            student_output = student(images)
            if 'vit' in cfg.model.network:
                teacher_output, teacher_attn = teacher(images[:2], return_attention=True)
            else:
                teacher_output = teacher(images[:2])
            loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                if logger: logger.error(f'loss is {loss.item()}, stopping training ------')
                sys.exit(1)
            if loss_am is not None: loss_am.update(loss.item())

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.unscale_(optimizer)

            for name, p in student.named_parameters():
                clip_grad = 3.0
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    clip_coef = clip_grad / (param_norm + 1e-6)
                    if clip_coef < 1:
                        p.grad.data.mul_(clip_coef)

                if epoch < cfg.train.lock_last_layer:
                    if 'last_layer' in name:
                        p.grad = None 

            if fp16_scaler is None:
                optimizer.step()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            with torch.no_grad():
                m = momentum_scheduler[steps]
                for param_s, param_t in zip(student.module.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            torch.cuda.synchronize()

            if writer:
                writer.add_scalar(f'train/loss', loss_am.val, steps)
                writer.add_scalar(f'train/lr', optimizer.param_groups[0]['lr'], steps)
                writer.add_scalar(f'train/wd', optimizer.param_groups[0]['weight_decay'], steps)

                if steps % cfg.image_writer_freq == 0:
                    img_arr = [img[0].cpu().numpy() for img in imgs]
                    for j in range(len(img_arr)):
                        img_arr[j] = img_arr[j].transpose([1, 2, 0])
                        img_arr[j] = (img_arr[j] * 0.5 + 0.5) * 255
                        img_arr[j] = img_arr[j].astype('uint8')

                    img1 = np.concatenate(img_arr[:2], axis=1)[:,:,::-1]
                    img2 = np.concatenate(img_arr[2:], axis=1)[:,:,::-1]
                    writer.add_image(f'image/global_image', img1, steps, dataformats='HWC')
                    writer.add_image(f'image/local_image', img2, steps, dataformats='HWC')

                    with torch.no_grad():
                        t_center = dino_loss.center.detach().cpu().numpy()[0]
                        t_out = teacher_output[0].detach().cpu().numpy()
                        s_out = student_output[-1].detach().cpu().numpy()

                        fig0 = plt.figure()
                        fig0.suptitle('teacher center')
                        plt.scatter(range(len(t_center)), t_center)
                        fig1 = plt.figure()
                        fig1.suptitle('teacher output')
                        plt.scatter(range(len(t_out)), t_out)
                        fig2 = plt.figure()
                        fig2.suptitle('student output')
                        plt.scatter(range(len(s_out)), s_out)
                        writer.add_figure(f'figure/teacher_center', fig0, steps)
                        writer.add_figure(f'figure/teacher', fig1, steps)
                        writer.add_figure(f'figure/student', fig2, steps)

                        if 'vit' in cfg.model.network:
                            attn_arr = teacher_attn.detach()[::imgs[0].shape[0]].cpu()
                            num, num_heads, _ = attn_arr.shape
                            img_size = cfg.model.input_size
                            patch_size = cfg.model.patch_size
                            attn_arr = attn_arr.reshape(num, num_heads, img_size[0]//patch_size, img_size[1]//patch_size)
                            attn_arr = nn.functional.interpolate(attn_arr, scale_factor=patch_size, mode='nearest').numpy()
                            
                            attn_arr_sum = []
                            for j in range(len(attn_arr)):
                                arr_sum = sum(attn_arr[j][k] * 1.0 / num_heads for k in range(num_heads))
                                arr_sum = arr_sum / arr_sum.max() * 255 
                                attn_arr_sum.append(arr_sum.astype('uint8'))

                            img3 = np.concatenate(attn_arr_sum, axis=1)
                            writer.add_image(f'image/attn_image', img3, steps, dataformats='HW')

            if steps % cfg.log_freq == 0 and logger:
                msg = f'[epoch{epoch}] steps {steps}, lr {optimizer.param_groups[0]["lr"]:.6f}, loss {loss_am.avg:.4f}'
                logger.info(msg)

        if len(face_folders) > 1 and RANK == 0:
            acc_arr = get_acc(teacher.backbone, cfg.model.network, face_dataloaders[:-1], face_dataloaders[-1], print_info=False)
            for i in range(len(face_folders) - 1):
                if logger: logger.info(f'[epoch{epoch}][megaface_test][{face_folders[i]}] acc {acc_arr[i]*100:.4f} ------')
                if writer: writer.add_scalar(f'train/test_{face_folders[i]}', acc_arr[i], epoch)

                if acc_arr[i] > best_acc[face_folders[i]] and RANK == 0:
                    best_acc[face_folders[i]] = acc_arr[i]
                    save_model(student, teacher, cfg.model.input_size, cfg.output, f'best_{face_folders[i]}', save_onnx=True)

    if RANK == 0: save_model(student, teacher, cfg.model.input_size, cfg.output, 'final', save_onnx=False)
    if writer: writer.close()
    
    distributed.destroy_process_group()
    if logger: logger.info('traing done ------')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True 
    cfg = get_config()
    main(cfg)
