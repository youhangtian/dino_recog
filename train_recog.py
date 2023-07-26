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

from src.utils import get_config_from_yaml, get_logger, AverageMeter 
from src.backbones import get_backbone 
from src.megaface_test import get_mega_dataloader, get_acc 

from src.recog.data import get_dataloader
from src.recog.losses import get_loss
from src.recog.partial_fc import PartialFC 
from src.recog.lr_scheduler import PolyScheduler

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
def save_model(model, path, name, save_onnx=False):
    save_path = os.path.join(path, f'{name}.pth')
    torch.save(model.module.state_dict(), save_path)

    if save_onnx:
        img = torch.randn(1, 3, 112, 112).to('cuda')
        torch.onnx.export(
            model, 
            img,
            os.path.join(path, f'{name}.onnx'),
            input_names=['input'],
            output_names=['features'],
            dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}},
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
    data_loader = get_dataloader(cfg.data, logger=logger)

    if logger: logger.info(f'get backbone: ------\n{cfg.model} ------')
    backbone = get_backbone(cfg.model, logger=logger).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[RANK],
        bucket_cap_mb=16
    )

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    if logger: logger.info(f'get partial fc, margin_loss {cfg.train.loss} ------')
    margin_loss = get_loss(cfg.train.loss, cfg.train.margin_list)
    num_classes = len(os.listdir(cfg.data.image_folder))
    module_partial_fc = PartialFC(
        margin_loss, 
        cfg.model.num_features, 
        num_classes, 
        cfg.data.sample_rate,
        cfg.model.fp16
    )
    module_partial_fc.train().cuda()

    if logger: logger.info(f'get optimizer: {cfg.train.optimizer} ------')
    if cfg.train.optimizer == 'sgd':
        opt = torch.optim.SGD(
            params=[{'params': backbone.parameters()}, {'params': module_partial_fc.parameters()}],
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer == 'adamw':
        opt = torch.optim.AdamW(
            params=[{'params': backbone.parameters()}, {'params': module_partial_fc.parameters()}],
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )
    else:
        raise ValueError(f'no such optimizer {cfg.train.optimizer}')

    if logger: logger.info(f'get lr scheduler ------')
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.train.lr,
        max_steps=cfg.train.num_epoch*len(data_loader),
        warmup_steps=cfg.train.warmup_epoch*len(data_loader)
    )

    face_dataloaders = []
    face_folders = cfg.data.megaface_face_folders.split(',')
    if logger: logger.info(f'get megaface dataloaders {face_folders} ------')

    for i in range(len(face_folders)):
        face_dataloader = get_mega_dataloader(cfg.data.megaface_data_root, face_folders[i], cfg.data.batch_size)
        face_dataloaders.append(face_dataloader)

    best_acc = {name: 0.0 for name in face_folders[:-1]}

    loss_am = AverageMeter() if RANK == 0 else None
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'tensorboard')) if RANK == 0 else None

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    steps = -1
    for epoch in range(-1*cfg.train.lock_epoch, cfg.train.num_epoch, 1):
        if logger: logger.info(f'epoch {epoch} begin ------')

        if epoch == 0:
            backbone.train()
            for param in backbone.parameters():
                param.requires_grad = True 

        if (epoch % cfg.train.save_epoch == 0) and (RANK == 0):
            save_model(backbone, cfg.output, f'epoch{epoch}', save_onnx=False)

        data_loader.sampler.set_epoch(epoch)

        for (imgs, labels) in tqdm(data_loader, f'[epoch{epoch}][rank{RANK}]'):
            steps += 1

            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            if 'vit' in cfg.model.network:
                embeddings, attn = backbone(imgs, return_attention=True)
            else:
                embeddings = backbone(imgs)

            loss = module_partial_fc(embeddings, labels)

            if not math.isfinite(loss.item()):
                logger.error(f'loss is {loss.item()}, stopping training ------')
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
            
            last_lr = lr_scheduler.get_last_lr()[0]

            with torch.no_grad():
                if loss_am: loss_am.update(loss.item())

                if writer:
                    writer.add_scalar(f'train/learning_rate', last_lr, steps)
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

                        for j in range(len(img_arr)):
                            img_arr[j] = img_arr[j].transpose([1, 2, 0])
                            img_arr[j] = img_arr[j] * std + mean 
                            img_arr[j] = img_arr[j] * 255

                        concat_img1 = np.concatenate((img_arr[0], img_arr[1])).astype('uint8')
                        concat_img2 = np.concatenate((img_arr[2], img_arr[3])).astype('uint8')
                        concat_img1 = concat_img1.copy()
                        concat_img2 = concat_img2.copy()
                        cv2.putText(concat_img1, f'{text_arr[0]:.2f}', (0, 122), 0, 1, (255, 255, 0))
                        cv2.putText(concat_img2, f'{text_arr[1]:.2f}', (0, 122), 0, 1, (255, 255, 0))
                        concat_img = np.concatenate((concat_img1, concat_img2), axis=1)[:,:,::-1]

                        writer.add_image(f'train/image', concat_img, steps, dataformats='HWC')

                        if 'vit' in cfg.model.network:
                            attn = attn.detach().cpu()
                            attn_arr = torch.cat([attn[idx].unsqueeze(0) for idx in [max_x, max_y, min_x, min_y]])
                            num, num_heads, _ = attn_arr.shape 
                            input_size = cfg.model.input_size 
                            patch_size = cfg.model.patch_size 
                            attn_arr = attn_arr.reshape(num, num_heads, input_size[0]//patch_size, input_size[1]//patch_size)
                            attn_arr = nn.functional.interpolate(attn_arr, scale_factor=patch_size, mode='nearest').numpy()

                            attn_arr_sum = []
                            for j in range(len(attn_arr)):
                                arr_sum = sum(attn_arr[j][k] * 1.0 / num_heads for k in range(num_heads))
                                arr_sum = arr_sum / arr_sum.max() * 255 
                                attn_arr_sum.append(arr_sum.astype('uint8'))

                            attn_img1 = np.concatenate((attn_arr_sum[0], attn_arr_sum[1]))
                            attn_img2 = np.concatenate((attn_arr_sum[2], attn_arr_sum[3]))
                            attn_concat_img = np.concatenate((attn_img1, attn_img2), axis=1)

                            writer.add_image(f'train/heatmap', attn_concat_img, steps, dataformats='HW')

            if (steps % cfg.log_freq == 0) and logger:
                msg = f'[epoch{epoch}] steps {steps}, lr {last_lr:.6f}, loss {loss_am.avg:.4f} ------'
                logger.info(msg)

            if epoch >=0: lr_scheduler.step()

        if len(face_folders) > 1 and RANK == 0:
            backbone.eval()
            acc_arr = get_acc(backbone, cfg.model.network, face_dataloaders[:-1], face_dataloaders[-1], print_info=False)
            for i in range(len(face_folders) - 1):
                if logger: logger.info(f'[epoch{epoch}][megaface_test][{face_folders[i]}] acc {acc_arr[i]*100:.4f} ------')
                if writer: writer.add_scalar(f'megaface/{face_folders[i]}', acc_arr[i], epoch)

                if acc_arr[i] > best_acc[face_folders[i]] and RANK == 0:
                    best_acc[face_folders[i]] = acc_arr[i] 
                    save_model(backbone, cfg.output, f'best_{face_folders[i]}', save_onnx=True)
            backbone.train()

    if RANK == 0:
        backbone.eval()
        save_model(backbone, cfg.output, f'backbone_final', save_onnx=False)
    if writer: writer.close()

    distributed.destroy_process_group()
    if logger: logger.info(f'training done, best acc arr: {best_acc} ------')

if __name__ == '__main__':
    torch.backends.cudnn.benckmark = True 
    cfg = get_config()
    main(cfg)
