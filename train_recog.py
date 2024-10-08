import os 
import sys 
import argparse 

import math
import cv2
import numpy as np 
from tqdm import tqdm

import torch 
from torch import nn
from torch import distributed 
from torch.utils.tensorboard import SummaryWriter 

from src.utils import get_config_from_yaml, get_logger, cosine_scheduler, AverageMeter 
from src.backbones import get_backbone 
from src.megaface_test import get_mega_dataloader, get_acc 

from src.recog.data import get_dataloader
from src.recog.circle_loss import CircleLoss 

def get_config():
    parser = argparse.ArgumentParser(description='training argument parser')
    parser.add_argument('-c', '--config_file', default='cfg_recog.yaml', type=str, help='config file')
    parser.add_argument('opts', help='modify config options from the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_config_from_yaml(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze
    return cfg

@torch.no_grad()
def save_model(model, path, name, input_size, save_onnx=False):
    save_path = os.path.join(path, f'{name}.pth')
    torch.save(model.module.state_dict(), save_path)

    if save_onnx:
        img = torch.randn(1, 3, input_size[0], input_size[1]).to('cuda')
        torch.onnx.export(
            model, 
            img,
            os.path.join(path, f'{name}.onnx'),
            input_names=['input'],
            output_names=['features', 'features_norm'],
            dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}, 'features_norm': {0: 'batch_size'}},
            opset_version=15
        )

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
    data_loader = get_dataloader(cfg.model.input_size, cfg.model.patch_size, cfg.data, logger=logger)

    if logger: logger.info(f'get backbone: ------\n{cfg.model} ------')
    backbone = get_backbone(cfg.model, logger=logger).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[RANK],
        bucket_cap_mb=16
    )

    if logger: logger.info(f'get partial fc, gamma: {cfg.train.gamma}, m: {cfg.train.m} ------')
    num_classes = len(os.listdir(cfg.data.image_folder))
    circle_loss = CircleLoss(
        cfg.train.gamma,
        cfg.train.m,  
        cfg.model.num_features, 
        num_classes, 
        cfg.train.sample_rate,
        cfg.model.fp16
    )
    circle_loss.train().cuda()

    circle_loss = torch.nn.parallel.DistributedDataParallel(
        module=circle_loss,
        broadcast_buffers=False,
        device_ids=[RANK],
        bucket_cap_mb=16
    )

    if logger: logger.info(f'get optimizer: {cfg.train.optimizer} ------')
    if cfg.train.optimizer == 'sgd':
        opt = torch.optim.SGD(
            params=[{'params': backbone.parameters()}, {'params': circle_loss.parameters()}],
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer == 'adamw':
        opt = torch.optim.AdamW(
            params=[{'params': backbone.parameters()}, {'params': circle_loss.parameters()}],
            lr=cfg.train.lr, 
            weight_decay=cfg.train.weight_decay
        )
    else:
        raise ValueError(f'no such optimizer {cfg.train.optimizer}')

    if logger: logger.info(f'get lr scheduler ------')
    lr_scheduler = cosine_scheduler(
        cfg.train.lr, 
        cfg.train.lr_end,
        cfg.train.epochs,
        len(data_loader),
        warmup_epochs=cfg.train.warmup_epochs,
        lock_epochs=cfg.train.lock_epochs
    )

    face_dataloaders = []
    face_folders = cfg.data.megaface_face_folders.split(',') if cfg.data.megaface_data_root else []
    if logger: logger.info(f'get megaface dataloaders {face_folders} ------')

    for i in range(len(face_folders)):
        face_dataloader = get_mega_dataloader(cfg.data.megaface_data_root, 
                                              face_folders[i], 
                                              cfg.data.batch_size, 
                                              cfg.model.input_size)
        face_dataloaders.append(face_dataloader)

    best_acc = {name: 0.0 for name in face_folders[:-1]}

    loss_am = AverageMeter() if RANK == 0 else None
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'tensorboard')) if RANK == 0 else None

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    steps = -1
    for epoch in range(-1*cfg.train.lock_epochs, cfg.train.epochs, 1):
        if logger: logger.info(f'epoch {epoch} begin ------')

        if (epoch % cfg.train.save_epoch == 0) and (RANK == 0):
            save_model(backbone, cfg.output, f'epoch{epoch}', cfg.model.input_size, save_onnx=False)

        data_loader.sampler.set_epoch(epoch)

        for (imgs, labels, masks) in tqdm(data_loader, f'[epoch{epoch}][rank{RANK}]'):
            steps += 1
            for i, param_group in enumerate(opt.param_groups):
                param_group['lr'] = lr_scheduler[steps]

            imgs = imgs.to('cuda')
            labels = labels.to('cuda')
            masks = masks.to('cuda')

            _, embeddings, _, attn = backbone(imgs, masks=masks, return_all=True)

            if epoch < 0:
                embeddings = embeddings.detach()

            loss = circle_loss(embeddings, labels)

            if not math.isfinite(loss.item()):
                if logger: logger.error(f'loss is {loss.item()}, stopping training ------')
                sys.exit(1)

            opt.zero_grad()
            if cfg.model.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step() 

            with torch.no_grad():
                if loss_am: loss_am.update(loss.item())

                if writer:
                    writer.add_scalar(f'train/learning_rate', opt.param_groups[0]['lr'], steps)
                    writer.add_scalar(f'train/loss', loss_am.val, steps)

                    if steps % cfg.image_writer_freq == 0:
                        batch_size = imgs.shape[0]
                        mat = nn.functional.linear(embeddings, embeddings)
                        sort_list = mat.flatten().argsort()
                        max_idx = sort_list[-batch_size-1].item()
                        min_idx = sort_list[0].item()
                        
                        max_x, max_y = max_idx // batch_size, max_idx % batch_size 
                        min_x, min_y = min_idx // batch_size, min_idx % batch_size 

                        img_arr = [imgs[idx].cpu().numpy() for idx in [max_x, max_y, min_x, min_y]]
                        text_arr = [mat[max_x][max_y].item(), mat[min_x][min_y].item()]

                        mean = [0.5, 0.5, 0.5]
                        std = [0.5, 0.5, 0.5]

                        input_size = cfg.model.input_size 
                        patch_size = cfg.model.patch_size

                        for j in range(len(img_arr)):
                            img_arr[j] = img_arr[j].transpose([1, 2, 0])
                            img_arr[j] = img_arr[j] * std + mean 
                            img_arr[j] = img_arr[j] * 255

                        concat_img = np.concatenate(img_arr, axis=1).astype('uint8')
                        concat_img = concat_img.copy()
                        cv2.putText(concat_img, f'{text_arr[0]:.2f}', (input_size[1] - 20, 20), 0, 1, (255, 255, 0))
                        cv2.putText(concat_img, f'{text_arr[1]:.2f}', (input_size[1] * 3 - 20, 20), 0, 1, (255, 255, 0))
                        writer.add_image(f'image/input_face', concat_img[:,:,::-1], steps, dataformats='HWC')

                        attn = attn.detach().cpu()
                        attn_arr = torch.cat([attn[idx].unsqueeze(0) for idx in [max_x, max_y, min_x, min_y]])
                        num, num_heads, _ = attn_arr.shape  
                        attn_arr = attn_arr.reshape(num, num_heads, input_size[0]//patch_size, input_size[1]//patch_size)
                        attn_arr = nn.functional.interpolate(attn_arr, scale_factor=patch_size, mode='nearest').numpy()

                        attn_arr_sum = []
                        for j in range(len(attn_arr)):
                            arr_sum = sum(attn_arr[j][k] * 1.0 / num_heads for k in range(num_heads))
                            arr_sum = arr_sum / arr_sum.max() * 255 
                            attn_arr_sum.append(arr_sum.astype('uint8'))

                        attn_concat_img = np.concatenate((attn_arr_sum), axis=1)
                        writer.add_image(f'image/heatmap', attn_concat_img, steps, dataformats='HW')

            if (steps % cfg.log_freq == 0) and logger:
                msg = f'[epoch{epoch}] steps {steps}, lr {opt.param_groups[0]["lr"]:.6f}, loss {loss_am.avg:.4f} ------'
                logger.info(msg)

        if len(face_folders) > 1 and RANK == 0:
            backbone.eval()
            acc_arr = get_acc(backbone, cfg.model.network, face_dataloaders[:-1], face_dataloaders[-1], print_info=False)
            for i in range(len(face_folders) - 1):
                if logger: logger.info(f'[epoch{epoch}][megaface_test][{face_folders[i]}] acc {acc_arr[i]*100:.2f}% ------')
                if writer: writer.add_scalar(f'train/test_{face_folders[i]}', acc_arr[i], epoch)

                if acc_arr[i] > best_acc[face_folders[i]] and RANK == 0:
                    best_acc[face_folders[i]] = acc_arr[i] 
                    save_model(backbone, cfg.output, f'best_{face_folders[i]}', cfg.model.input_size, save_onnx=True)
            backbone.train()

    if RANK == 0:
        backbone.eval()
        save_model(backbone, cfg.output, f'backbone_final', cfg.model.input_size, save_onnx=False)
    if writer: writer.close()

    distributed.destroy_process_group()
    if logger: logger.info(f'training done, best acc arr: {best_acc} ------')

if __name__ == '__main__':
    torch.backends.cudnn.benckmark = True 
    cfg = get_config()
    main(cfg)
