from PIL import Image
import torch
from torch import nn
import numpy as np
import sys
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import make_grid, save_image

transforms = Compose([
    ToTensor(),
])

with Image.open("/home/defonte/PersonReid/synth-reid/2602752943_sd007c1_0464.jpeg") as im:
    im = transforms(im)


# features = torch.rand((32, 2048, 32, 16))
features = im.reshape(1, 3, 128, 64)
print(features.shape)
permutations = np.load("./perm_31.npy")
print(permutations.shape)
grid_size_h = 2
grid_size_v = 4


def get_tile(feature, index):
    if (feature.shape[-2] % grid_size_v != 0 or
            feature.shape[-1] % grid_size_h != 0):
        raise ValueError("The vertical and horizontal numger of \
                            grids must be a multiple of the features shape")

    h = int(feature.shape[-2] / grid_size_v)
    w = int(feature.shape[-1] / grid_size_h)

    x = index % grid_size_h
    y = int(index / grid_size_h)
    tile = feature[:, y * h:(y + 1) * h, x * w:(x + 1) * w]
    return tile


feature_tiles = []
for feature in features:
    tiles = []
    for n in range(grid_size_h * grid_size_v):
        tiles.append(get_tile(feature, n))
    tiles = torch.stack(tiles, dim=0)
    index = np.random.choice(permutations.shape[0])
    permutation = permutations[index]
    print(permutation)
    tiles = tiles[permutation]
    # tiles = tiles.permute(1,0,2,3)
    tiles = make_grid(tiles, nrow=grid_size_h, padding=0)
    # save_image(tiles, './img1.png')
    # sys.exit()
    feature_tiles.append(tiles)

feature_tiles = torch.stack(feature_tiles, dim=0)

save_image(feature_tiles[0], './img1.png')
