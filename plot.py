import os
import matplotlib.pyplot as plt
from PIL import Image

files = ["original.png", "scaled.png", "espcn.png", "edsr.png", "resrgan.png"]
plt.figure(figsize=(30, 30))
for i, file in enumerate(files):
    plt.subplot(1, 5, i + 1)
    img = Image.open(os.path.join("results", file))
    plt.imshow(img)
    plt.axis("off")
    plt.title(file.split(".")[0])
plt.show()
