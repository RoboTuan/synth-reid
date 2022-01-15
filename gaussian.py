from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose
import matplotlib.pylab as plt
import sys


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=(256, 128), std=(128, 128)):
    """Returns a 2D Gaussian kernel array."""
    gkern1d1 = gaussian_fn(kernlen[0], std=std[0])
    gkern1d2 = gaussian_fn(kernlen[1], std=std[1])
    gkern2d = torch.outer(gkern1d1, gkern1d2)
    gkern2d = gkern2d.expand(3, kernlen[0], kernlen[1])
    return gkern2d


# Generate random matrix and multiply the kernel by it

transforms = Compose([
    ToTensor(),
])

with Image.open("/home/defonte/PersonReid/synth-reid/2602752943_sd007c1_0464.jpeg") as img:
    img = transforms(img)
gaussian_filter = gkern((128, 64), std=(32, 16))
print(gaussian_filter.shape)

ax = []
f = plt.figure(figsize=(12, 5))
ax.append(f.add_subplot(131))
ax.append(f.add_subplot(132))
ax.append(f.add_subplot(133))
ax[0].imshow(img.permute(1, 2, 0))
ax[1].imshow(gaussian_filter.permute(1, 2, 0))
ax[2].imshow((img * gaussian_filter).permute(1, 2, 0))
plt.savefig("dummy_fig.png")
