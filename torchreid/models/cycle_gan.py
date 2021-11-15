import torch.nn as nn
from torchreid.utils import gan_weights_init

class cycleGenerator(nn.Module):
    def __init__(self, generator_S2R, generator_R2S):
        super(cycleGenerator, self).__init__()
        self.generator_S2R = generator_S2R
        self.generator_S2R.apply(gan_weights_init)

        self.generator_R2S = generator_R2S
        self.generator_R2S.apply(gan_weights_init)


def make_generators(generator_S2R, generator_R2S):
    generator = cycleGenerator(generator_S2R, generator_R2S)
    return generator


def make_discriminator(discriminator):
    discriminator = discriminator.apply(gan_weights_init)
    return discriminator
