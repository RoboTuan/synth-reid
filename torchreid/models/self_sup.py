import torch
from torch import nn
import numpy as np
from torchvision.utils import make_grid
import collections
import sys


class SelfSup(nn.Module):
    def __init__(self,
                 backbone,
                 grid_size_h,
                 grid_size_v,
                 permutations='./perm_31.npy',
                 **kwargs):
        super(SelfSup, self).__init__()

        self.backbone = backbone
        # The grid size is grid_size_h x grid_size_v such as 2x4
        self.grid_size_h = grid_size_h
        self.grid_size_v = grid_size_v
        self.permutations = np.load(permutations)
        self.num_jig_classes = self.permutations.shape[0]

        # TODO: put bottleneck for bigger models
        jig_feature_extractor = nn.Conv2d(self.backbone.inplanes,
                                          self.backbone.inplanes,
                                          kernel_size=3,
                                          padding=1)
        global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        jig_classifier = nn.Linear(self.backbone.inplanes, self.num_jig_classes)

        self.jig_saw_puzzle = nn.Sequential(collections.OrderedDict(
                                            [
                                                ("feat_extractor", jig_feature_extractor),
                                                ("global_avgpool", global_avgpool),
                                                ("flatten", flatten),
                                                ("classifier", jig_classifier)
                                            ]
                                            ))

    def forward(self, x):
        outputs = self.backbone.forward(x)
        # print(self.training)
        if not self.training:
            return outputs

        if self.backbone.loss == 'softmax':  # Softmax
            output_class, features = outputs
        elif self.backbone.loss == 'triplet':  # Triplet
            output_class, avg_pool, features = outputs
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        feature_tiles, true_labels = self.permute_tiles(features)
        feature_tiles = torch.stack(feature_tiles, dim=0)
        # print("original: ", features[0, 0, :, :])
        # print("permuted: ", feature_tiles[0, 0, :, :])
        # print(features.shape, feature_tiles.shape)
        # print(true_labels)
        # sys.exit()
        jig_outputs = self.jig_saw_puzzle(feature_tiles)

        if self.backbone.loss == 'softmax':
            return output_class, jig_outputs, true_labels
        else:  # check for loss error was already made before
            return output_class, avg_pool, jig_outputs, true_labels

    def permute_tiles(self, features):
        feature_tiles = []
        true_labels = []
        for feature in features:
            tiles = []
            for n in range(self.grid_size_h * self.grid_size_v):
                tiles.append(self.get_tile(feature, n))
            tiles = torch.stack(tiles, dim=0)
            index = np.random.choice(self.permutations.shape[0])
            true_labels.append(index)
            permutation = self.permutations[index]
            tiles = tiles[permutation]
            tiles = make_grid(tiles, nrow=self.grid_size_h, padding=0)
            feature_tiles.append(tiles)
        return feature_tiles, torch.LongTensor(true_labels)

    def get_tile(self, feature, index):
        if (feature.shape[-2] % self.grid_size_v != 0 or
                feature.shape[-1] % self.grid_size_h != 0):
            raise ValueError("The vertical and horizontal numger of \
                              grids must be a multiple of the features shape")

        h = int(feature.shape[-2] / self.grid_size_v)
        w = int(feature.shape[-1] / self.grid_size_h)

        x = index % self.grid_size_h
        y = int(index / self.grid_size_h)
        tile = feature[:, y * h:(y + 1) * h, x * w:(x + 1) * w]
        return tile
