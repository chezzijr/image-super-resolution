import torch
import json
from model import espcn, edsr, EarlyStopping
from os.path import join, exists
from os import getcwd
from utils import hr_transform, lr_transform

modules = [espcn, edsr]
upscale_factor = 4
weights_path = join(getcwd(), "weights")
stats_path = join(getcwd(), "stats")
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = join(getcwd(), "checkpoint")

def train():
    for module in modules:
        name = module.__name__.split(".")[-1]
        callback = EarlyStopping(patience=5) 
        
        # checkpoint from latest epoch if any
        model_checkpoint_path = join(checkpoint_path, f"{name}_x{upscale_factor}_checkpoint.pth")
        checkpoint = None
        if exists(model_checkpoint_path):
            checkpoint = torch.load(model_checkpoint_path)

        # if training got interrupted, return checkpoint from latest epoch and return finished = false
        # if we want to resume training, we can pass in the checkpoint as argument
        model, checkpoint, finished = module.train_model(
            join(getcwd(), "datasets", "train"),
            join(getcwd(), "datasets", "validate"),
            hr_transform,
            lr_transform,
            device=device,
            upscale_factor=upscale_factor,
            lr=1e-4,
            callback=callback,
            checkpoint=checkpoint
        )

        if finished:
            # save model weights and stats
            torch.save(model.state_dict(), join(weights_path, f"{name}_x{upscale_factor}.pth"))
            with open(join(stats_path, f"{name}_x{upscale_factor}_stats.json"), "w") as f:
                json.dump([stat.to_dict() for stat in checkpoint.stats], f)
        elif len(checkpoint.stats) > 0: # At least one epoch was completed
            torch.save(checkpoint, model_checkpoint_path)

if __name__ == "__main__":
    train()
