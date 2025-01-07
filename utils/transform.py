# type: ignore
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    ToPILImage,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomCrop,
)
from PIL.Image import Image, Resampling
import random


def random_crop(image: Image, divisible_by: int):
    w, h = image.size
    dw = w % divisible_by
    dh = h % divisible_by
    x = random.randint(0, dw)
    y = random.randint(0, dh)
    new_w = w - dw
    new_h = h - dh
    return image.crop((x, y, x + new_w, y + new_h))


def resize(im: Image, upscale_factor: int):
    w, h = im.size
    return im.resize(
        (w // upscale_factor, h // upscale_factor), resample=Resampling.BICUBIC
    )

def hr_transform(img: torch.Tensor, upscale_factor: int):
    img: Image = ToPILImage()(img)
    img = random_crop(img, upscale_factor)
    # randomly rotate the image by 90, or 270 degrees
    if random.random() > 0.5:
        degree = random.choice([90, 270])
        img = img.rotate(degree, expand=True)

    # random crop image but w and h should be divisible by upscale_factor
    # cropped image still maintains the same aspect ratio and at least a quarter of the original image
    w, h = img.size
    crop_w = random.randrange(w, w // 2 - 1, -upscale_factor)
    crop_h = random.randrange(h, h // 2 - 1, -upscale_factor)

    return Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomCrop((crop_h, crop_w)),
        ToTensor(),
    ])(img)

def lr_transform(img: torch.Tensor, upscale_factor: int):
    img = ToPILImage()(img)
    img = resize(img, upscale_factor)
    return ToTensor()(img)
