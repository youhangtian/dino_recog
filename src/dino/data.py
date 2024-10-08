import os 
import math 
import random 
import torch 
import cv2 
import numpy as np 
from PIL import Image, ImageFilter, ImageOps

from torchvision import datasets, transforms 

from torch.utils.data import DistributedSampler, DataLoader 
import torch.distributed as dist 

from .utils import make_square, rotate_img


def get_dataloader(patch_size, input_size, cfg, shuffle=True, drop_last=True):
    seed = setup_seed()

    global_resize = (input_size[0], input_size[1])
    local_resize = (input_size[0] // 2, input_size[1] // 2)

    transform = DataAugmentationDINO(
        global_resize,
        local_resize,
        global_crops_scale=cfg.global_crops_scale,
        local_crops_scale=cfg.local_crops_scale,
        local_crops_number=cfg.local_crops_number
    )

    dataset = CustomImageFolderDataset(cfg, transform, patch_size, input_size)

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


class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self, cfg, transform, patch_size, input_size):
        super(CustomImageFolderDataset, self).__init__(cfg.image_folder, transform)
        self.root = cfg.image_folder
        self.transform = transform 

        self.patch_size = patch_size 
        self.input_size = input_size
        self.cfg = cfg  

        min_aspect = 0.3
        max_aspect = 1.0 / 0.3
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch 

    def get_mask_ratio(self):        
        assert self.cfg.mask_ratio > self.cfg.mask_ratio_var 

        mask_ratio = random.uniform(
            self.cfg.mask_ratio - self.cfg.mask_ratio_var,
            self.cfg.mask_ratio + self.cfg.mask_ratio_var
        ) if self.cfg.mask_ratio_var > 0 else self.cfg.mask_ratio 

        return mask_ratio 

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_bgr = cv2.imread(path)
        if self.cfg.make_square: img_bgr = make_square(img_bgr)
        if self.cfg.rotate_degree: img_bgr = rotate_img(img_bgr, self.cfg.rotate_degree)

        img = Image.fromarray(img_bgr.astype(np.uint8))
        sample = self.transform(img)

        H, W = sample[0].shape[1] // self.patch_size, sample[0].shape[2] // self.patch_size 
        high = self.get_mask_ratio() * H * W 

        if self.cfg.mask_shape == 'block':
            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0 
            while mask_count < high:
                max_mask_patches = high - mask_count 

                delta = 0 
                for attempt in range(10):
                    low = (min(H, W) // 3) ** 2 
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top:top+h, left:left+w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1 

                    if delta > 0:
                        break 

                if delta == 0:
                    break 
                else:
                    mask_count += delta 
        else:
            mask = np.hstack([np.zeros(H * W - int(high)), np.ones(int(high))]).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(H, W)

        masks = [mask.flatten() for _ in range(len(sample))]

        return sample, target, masks 
    

class DataAugmentationDINO(object):
    def __init__(self,
                 global_resize,
                 local_resize,
                 global_crops_scale=(0.5, 1.0),
                 local_crops_scale=(0.05, 0.5),
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

        global_crops_scale = (global_crops_scale[0], global_crops_scale[1])
        local_crops_scale = (local_crops_scale[0], local_crops_scale[1])
        
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
    