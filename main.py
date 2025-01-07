import torch
import json
from model import espcn, edsr, EarlyStopping
from os.path import join, exists
from os import getcwd
from utils import hr_transform, lr_transform
from torchvision.io import read_image
from torchvision.transforms import ToPILImage

modules = [espcn]
upscale_factor = 4
weights_path = join(getcwd(), "weights")
stats_path = join(getcwd(), "stats")
checkpoint_path = join(getcwd(), "checkpoints")

def train():
    for module in modules:
        name = module.__name__.split(".")[-1]
        callback = EarlyStopping(patience=5) 
        checkpoint_path = join(checkpoint_path, f"{name}_x{upscale_factor}_checkpoint.pth")
        if exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = None

        model, checkpoint, finished = module.train_model(
            join(getcwd(), "datasets", "train"),
            join(getcwd(), "datasets", "validate"),
            hr_transform,
            lr_transform,
            device="cuda" if torch.cuda.is_available() else "cpu",
            upscale_factor=upscale_factor,
            lr=1e-4,
            callback=callback,
            checkpoint=checkpoint
        )

        if finished:
            torch.save(model.state_dict(), join(weights_path, f"{name}_x{upscale_factor}.pth"))
            with open(join(stats_path, f"{name}_x{upscale_factor}_stats.json"), "w") as f:
                json.dump([stat.to_dict() for stat in checkpoint.stats], f)
        else:
            torch.save(checkpoint, checkpoint_path)

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    espcn_model = espcn.ESPCN(scale_factor=upscale_factor).to(device)
    edsr_model = edsr.EDSR(scale_factor=upscale_factor).to(device)

    espcn_model.load_state_dict(torch.load(join(weights_path, f"espcn_x{upscale_factor}.pth")))
    edsr_model.load_state_dict(torch.load(join(weights_path, f"edsr_x{upscale_factor}.pth")))

    espcn_model.eval()
    edsr_model.eval()

    test_path = join(getcwd(), "datasets", "real")

    path = join(test_path, "test.jpg")
    input_tensor = read_image(path).to(device)
    predict_espcn = espcn_model.predict_image(input_tensor)
    predict_edsr = edsr_model.predict_image(input_tensor)

    img_espcn = ToPILImage()(predict_espcn)
    img_edsr = ToPILImage()(predict_edsr)

def compare():
    with open(join(stats_path, "espcn_x4_stats.json"), "r") as f:
        espcn_stats = json.load(f)
    with open(join(stats_path, "edsr_x4_stats.json"), "r") as f:
        edsr_stats = json.load(f)
    print("Highest PSNR for ESPCN: ", max(s['psnr'] for s in espcn_stats))
    print("Highest PSNR for EDSR: ", max(s['psnr'] for s in edsr_stats))


if __name__ == "__main__":
    train()
