import torch
import json
import os
from model import espcn, edsr, EarlyStopping
from os.path import join, exists
from os import getcwd
from utils import hr_transform, lr_transform
import kagglehub
import shutil

modules = [espcn, edsr]
upscale_factor = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = join(getcwd(), "weights")
stats_path = join(getcwd(), "stats")
checkpoint_path = join(getcwd(), "checkpoint")
# ensure paths
os.makedirs(weights_path, exist_ok=True)
os.makedirs(stats_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

def prepare_dataset():
    if exists(join(getcwd(), "datasets", "train")) and exists(join(getcwd(), "datasets", "validate")):
        return

    # Download latest version
    path = kagglehub.dataset_download("sharansmenon/div2k")
    print("Path to dataset files:", path) 

    os.makedirs(join(getcwd(), "datasets"), exist_ok=True)
    os.makedirs(join(getcwd(), "datasets", "train"), exist_ok=True)
    os.makedirs(join(getcwd(), "datasets", "validate"), exist_ok=True)
    
    # move all files
    kaggle_train_files = join(path, "DIV2K_train_HR", "DIV2K_train_HR")
    kaggle_validate_files = join(path, "DIV2K_valid_HR", "DIV2K_valid_HR")
    train_dir = join(getcwd(), "datasets", "train")
    validate_dir = join(getcwd(), "datasets", "validate")
    train_files = os.listdir(kaggle_train_files)
    validate_files = os.listdir(kaggle_validate_files)
    for train_file in train_files:
        shutil.move(join(kaggle_train_files, train_file), join(getcwd(), train_dir))

    for validate_file in validate_files:
        shutil.move(join(kaggle_validate_files, validate_file), join(getcwd(), validate_dir))

    # empty cache
    shutil.rmtree(path)

def train():
    for module in modules:
        name = module.__name__.split(".")[-1]
        callback = EarlyStopping(patience=5) 
        
        # checkpoint from latest epoch if any
        # ensure checkpoint path
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
    # train()
    prepare_dataset()
