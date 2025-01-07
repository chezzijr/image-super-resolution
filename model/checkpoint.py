import torch
from dataclasses import dataclass
from .early_stopping import EarlyStopping

@dataclass 
class Stats:
    train_loss: float
    val_loss: float
    psnr: float

    def to_dict(self):
        return {"train_loss": self.train_loss, "val_loss": self.val_loss, "psnr": self.psnr}

@dataclass
class Checkpoint:
    stats: list[Stats] # List of Stats objects up to the current epoch
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    early_stopping: EarlyStopping | None
