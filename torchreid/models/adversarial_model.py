import torch.nn as nn
from torchreid.utils import weights_init_kaiming
from .backbones import Discriminator, Generator, MLP, Id_Net


def make_generator():
    generator = Generator()
    return generator.apply(weights_init_kaiming)


def make_discriminator():
    discriminator = Discriminator()
    return discriminator.apply(weights_init_kaiming)


# def make_mlp(use_mlp, nc, use_gpu, optim, lr, adam_beta1, step_size):
#     mlp = MLP(use_mlp, nc, use_gpu, optim, lr, adam_beta1, step_size)
#     return mlp


def make_id_net(in_planes, num_classes):
    id_net = Id_Net(in_planes, num_classes)
    return id_net.apply(weights_init_kaiming)
