import os
import math
import random 
import numpy as np
import torch 
from PIL import Image, ImageFilter, ImageOps 

from torchvision import datasets, transforms

from torch.utils.data import DistributedSampler, DataLoader

import torch.distributed as dist 


def get_dataloader(cfg, shuffle=True, drop_last=True):
    seed = setup_seed()

    transform = transforms.Compose([
        ConvertColor(), 
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.Resize((cfg.resize[0] // 2, cfg.resize[1] // 2)),
        ]), p=0.5), 
        transforms.Resize((cfg.resize[0], cfg.resize[1])), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)
        ]), p=0.5), 
        # transforms.RandomGrayscale(p=0.1), 
        GaussianBlur(0.5), 
        # Solarization(0.2), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        transforms.RandomErasing(p=0.5, scale=(0.1, 0.5)), 
    ])

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
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = cuda_deterministic 
    torch.backends.cudnn.benchmark = not cuda_deterministic 

    return seed


class ConvertColor(object):
    def __call__(self, img):
        img = np.array(img)[:,:,::-1]
        return Image.fromarray(np.uint8(img))
    

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
    def __init__(
        self,
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

