import torch
import tqdm
import math
from torch import nn
from .dataset import CustomDataset
from .early_stopping import EarlyStopping
from .checkpoint import Checkpoint, Stats


class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=3):
        super(ESPCN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, num_channels * scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x

    def predict_image(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            x = self(x)
        x = x.squeeze(0)
        x = x * 255.0
        return x.byte()


def train_model(
    train_path: str,
    val_path: str,
    hr_transform,
    lr_transform,
    device,
    lr=1e-4,
    upscale_factor=4,
    epochs=50,
    callback=EarlyStopping(patience=3),
    checkpoint: Checkpoint | None = None,
):
    train = CustomDataset(
        train_path,
        upscale_factor,
        hr_transform,
        lr_transform,
        device,
    )
    validate = CustomDataset(
        val_path,
        upscale_factor,
        hr_transform,
        lr_transform,
        device,
    )

    model = ESPCN(upscale_factor)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # print model summary
    print(model)
    print("Model has", sum(p.numel() for p in model.parameters()), "parameters.")

    # Training
    if checkpoint is None:
        print("Starting training from scratch.")
        checkpoint = Checkpoint(
            stats=[],
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            early_stopping=callback,
        )
    else:
        print("Resuming training from last checkpoint.")
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        callback = callback if checkpoint.early_stopping is None else checkpoint.early_stopping

    finished = False
    for epoch in range(len(checkpoint.stats), epochs):
        try:
            model.train()
            train.shuffle()
            total_loss = 0
            with tqdm.tqdm(train, unit="image") as t:
                for i, (image, target) in enumerate(t):
                    optimizer.zero_grad()
                    output = model(image)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    t.set_description(
                        f"Epoch {epoch + 1}/{epochs}, AVG Loss: {total_loss / (i + 1):.9f}"
                    )
            avg_loss = total_loss / len(train)

            model.eval()
            validate.shuffle()
            total_val_loss = 0
            psnr = 0
            for image, target in validate:
                output = model(image)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                psnr += 10 * math.log10(1.0 / loss.item())
            avg_psnr = psnr / len(validate)
            avg_val_loss = total_val_loss / len(validate)
            print(f"Validation loss: {avg_loss:.9f}, PSNR: {avg_psnr:.6f}")

            checkpoint.stats.append(Stats(avg_loss, avg_val_loss, avg_psnr))
            checkpoint.model_state_dict = model.state_dict()
            checkpoint.optimizer_state_dict = optimizer.state_dict()

            if callback(avg_loss):
                print(
                    f"Early stopping as validation loss did not improve in {callback.patience} epochs."
                )
                finished = True
                break
            checkpoint.early_stopping = callback

        except KeyboardInterrupt:
            print("Training interrupted. Returning checkpoint from last epoch.")
            break
    else:
        finished = True
        print("Training completed succesfully.")

    # Save model
    return model, checkpoint, finished
