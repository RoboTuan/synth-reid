import torch.nn as nn


class cycleGenerator(nn.Module):
    def __init__(self, generator_S2R, generator_R2S):
        super(cycleGenerator, self).__init__()
        self.generator_S2R = generator_S2R
        self.generator_R2S = generator_R2S


def make_generators(generator_S2R, generator_R2S):
    generator = cycleGenerator(generator_S2R, generator_R2S)
    return generator


def make_discriminator(discriminator):
    return discriminator
