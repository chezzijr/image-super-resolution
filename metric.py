import matplotlib.pyplot as plt
import json
from os.path import join

stats_path = "stats"
stats_path = join(stats_path, "edsr_x4_stats.json")
# edsr_stats_path = join(stats_path, "edsr_x4_stats.json")

with open(stats_path, "r") as f:
    stats = json.load(f)
# stats consist of list of dictionaries, each dictionary contains training loss, validation loss, and psnr

training_losses = [s['train_loss'] for s in stats]
validation_losses = [s['val_loss'] for s in stats]

# plot both training and validation loss in the same graph
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

psnrs = [s['psnr'] for s in stats]
plt.plot(psnrs)
plt.title("PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.show()

