import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, img_dir, upscale_factor, hr_transform, lr_transform, device):
        self.img_dir = img_dir
        self.upscale_factor = upscale_factor
        self.imgs = os.listdir(img_dir)

        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = read_image(img_path).to(self.device)
        hr = self.hr_transform(img, self.upscale_factor)
        lr = self.lr_transform(hr, self.upscale_factor)
        hr = torch.unsqueeze(hr, 0)
        lr = torch.unsqueeze(lr, 0)
        hr = hr.to(self.device)
        lr = lr.to(self.device)
        return lr, hr

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def shuffle(self):
        random.shuffle(self.imgs)
