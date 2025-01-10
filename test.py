import torch
import json
import matplotlib.pyplot as plt
import os
from model import espcn, edsr, EarlyStopping
from os.path import join, exists
from os import getcwd
from utils import hr_transform, lr_transform
from torchvision.io import read_image
from torchvision.transforms import ToPILImage, ToTensor
from RealESRGAN import RealESRGAN
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
upscale_factor = 4
weights_path = join(getcwd(), "weights")
def test(img_path):

    espcn_model = espcn.ESPCN(scale_factor=upscale_factor).to(device)
    edsr_model = edsr.EDSR(scale_factor=upscale_factor).to(device)
    resrgan_model = RealESRGAN(device=device, scale=upscale_factor)

    espcn_model.load_state_dict(torch.load(join(weights_path, f"espcn_x{upscale_factor}.pth")))
    edsr_model.load_state_dict(torch.load(join(weights_path, f"edsr_x{upscale_factor}.pth")))
    resrgan_model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    # ensure that there are results folder
    os.makedirs("results", exist_ok=True)
    save_path = join(getcwd(), "results")

    img = Image.open(img_path).convert("RGB")
    img.save(join(save_path, "original.png"))
    scaled = img.resize((img.width // upscale_factor, img.height // upscale_factor), Image.Resampling.BICUBIC)
    scaled.save(join(save_path, "scaled.png"))

    input_tensor = ToTensor()(scaled).to(device)
    predict_espcn = ToPILImage()(espcn_model.predict_image(input_tensor))
    predict_edsr = ToPILImage()(edsr_model.predict_image(input_tensor))
    predict_resrgan = resrgan_model.predict(scaled)

    predict_espcn.save(join(save_path, "espcn.png"))
    predict_edsr.save(join(save_path, "edsr.png"))
    predict_resrgan.save(join(save_path, "resrgan.png"))

if __name__ == "__main__":
    test("datasets/test/0801.png")

