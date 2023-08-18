import os
import math
import random 
import torch 
import cv2 
import numpy as np
from PIL import Image

from torchvision import datasets, transforms

from torch.utils.data import DistributedSampler, DataLoader

import torch.distributed as dist 


def get_dataloader(input_size, patch_size, cfg, shuffle=True, drop_last=True, logger=None):
    seed = setup_seed()
    if logger: logger.info(f'data loader seed {seed} ------')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)
        ]), p=0.5), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])

    dataset = CustomImageFolderDataset(cfg, transform, input_size, patch_size)

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

    
class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self, cfg, transform, input_size, patch_size):
        super(CustomImageFolderDataset, self).__init__(cfg.image_folder, transform)
        self.root = cfg.image_folder 
        self.transform = transform

        self.input_size = input_size
        self.patch_size = patch_size
        self.cfg = cfg

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_bgr = cv2.imread(path)

        if np.random.random() < 0.5:
            img_shape = img_bgr.shape 
            side_ratio = np.random.uniform(0.3, 0.7)
            new_shape = (int(side_ratio * img_shape[1]), int(side_ratio * img_shape[0]))
            interpolation = np.random.choice(
                [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
            )
            img_bgr = cv2.resize(img_bgr, new_shape, interpolation=interpolation)

        img_bgr = cv2.resize(img_bgr, (self.input_size[1], self.input_size[0]))

        img = Image.fromarray(img_bgr.astype(np.uint8))
        sample = self.transform(img)

        H, W = self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size
        high = int(np.random.uniform(0.0, 0.5) * H * W)
        mask = np.hstack([np.zeros(H * W - high), np.ones(high)]).astype(bool)
        np.random.shuffle(mask)

        return sample, target, mask 


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

