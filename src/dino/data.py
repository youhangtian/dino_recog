import os 
import math 
import random 
import torch 
from PIL import ImageFilter, ImageOps
import numpy as np 

from torchvision import datasets, transforms 

from torch.utils.data import DistributedSampler, DataLoader 

import torch.distributed as dist 


def get_dataloader(cfg, shuffle=True, drop_last=True):
    seed = setup_seed()

    global_resize = (cfg.resize[0], cfg.resize[1])
    local_resize = (cfg.resize[0] // 2, cfg.resize[1] // 2)

    transform = DataAugmentationDINO(
        global_resize,
        local_resize,
        local_crops_number=cfg.local_crops_number
    )
    dataset = datasets.ImageFolder(cfg.image_folder, transform=transform)

    rank, world_size = get_dist_info()
    sampler = DistSampler(dataset, seed, world_size, rank, shuffle)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        drop_last=drop_last
    )
    return dataloader


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1 
    return rank, world_size 


def setup_seed(device='cuda', cuda_deterministic=False):
    seed = np.random.randint(10000)

    rank, world_size = get_dist_info()

    if world_size > 1:
        if rank == 0:
            random_num = torch.tensor(seed, dtype=torch.int32, device=device)
        else:
            random_num = torch.tensor(0, dtype=torch.int32, device=device)

        dist.broadcast(random_num, src=0)
        seed = random_num.item()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = cuda_deterministic 
    torch.backends.cudnn.benchmark = not cuda_deterministic

    return seed 


class DataAugmentationDINO(object):
    def __init__(self,
                 global_resize,
                 local_resize,
                 global_crops_scale=(0.8, 1.0),
                 local_crops_scale=(0.05, 0.8),
                 local_crops_number=8):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        global_ratio = (global_resize[1] * 0.75 / global_resize[0], 
                        global_resize[1] * 1.3333 / global_resize[0])
        
        local_ratio = (local_resize[1] * 0.75 / local_resize[0],
                       local_resize[1] * 1.3333 / local_resize[0])
        
        self.global_trans1 = transforms.Compose([
            transforms.RandomResizedCrop(global_resize, 
                                         scale=global_crops_scale, 
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         ratio=global_ratio),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])

        self.global_trans2 = transforms.Compose([
            transforms.RandomResizedCrop(global_resize, 
                                         scale=global_crops_scale, 
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         ratio=global_ratio),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number 
        self.local_trans = transforms.Compose([
            transforms.RandomResizedCrop(local_resize, 
                                         scale=local_crops_scale, 
                                         interpolation=transforms.InterpolationMode.BICUBIC,
                                         ratio=local_ratio),
            flip_and_color_jitter,
            GaussianBlur(0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_trans1(image))
        crops.append(self.global_trans2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_trans(image))
        return crops 


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.p = p 
        self.radius_min = radius_min 
        self.radius_max = radius_max 

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(
                ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        else:
            return img
        

class Solarization(object):
    def __init__(self, p):
        self.p = p 

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class DistSampler(DistributedSampler):
    def __init__(self,
                 dataset,
                 seed,
                 num_replicas,
                 rank,
                 shuffle):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = seed 

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = (indices * math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size 

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples 

        return iter(indices)
    