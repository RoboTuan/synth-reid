from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid.models.backbones.resnet import BasicBlock, conv1x1, Bottleneck
import sys


# norm = functools.partial(nn.InstanceNorm2d, affine=False)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, k=8):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.C_tilde = self.chanel_in // k
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.C_tilde, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.C_tilde, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.C_tilde, kernel_size=1)
        self.self_att = nn.Conv2d(in_channels=self.C_tilde, out_channels=self.chanel_in, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, _, H, W = x.size()  # N = W * H
        # print(x.size())
        f_x = self.query_conv(x).flatten(2).transpose(1, 2)  # B x N x C//8
        # print(f_x.size())
        g_x = self.key_conv(x).flatten(2)  # B x C//8 x N
        # print(g_x.size())
        h_x = self.value_conv(x).flatten(2)  # B X C//8 X N
        # print(h_x.size())

        s = torch.bmm(f_x, g_x)  # transpose check B x N x N
        # print(s.size())
        beta = self.softmax(s)  # B x N x N
        # print(beta.size())

        v = torch.bmm(h_x, beta)  # B x C//8 x N
        # print(v.size())
        v = v.view(B, self.C_tilde, H, W)
        # print(v.size())

        o = self.self_att(v)
        # print(o.size())
        y = self.gamma * o + x
        # print(self.gamma)
        # print(y.size())
        # sys.exit()

        return y


def check_norm(norm, out_dim):
    if norm == "InstanceNorm":
        norm = functools.partial(nn.InstanceNorm2d, num_features=out_dim, affine=False)
    elif norm == "BatchNorm":
        norm = functools.partial(nn.BatchNorm2d, num_features=out_dim, affine=False)
    elif norm == "SpectralNorm":
        norm = functools.partial(nn.utils.spectral_norm)
    else:
        raise NotImplementedError
    return norm


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm="InstanceNorm", activation=nn.ReLU):

    norm = check_norm(norm, out_dim)
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(),
        activation())


def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm="InstanceNorm", activation=nn.ReLU):

    norm = check_norm(norm, out_dim)
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(),
        activation())


class Discriminator(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_bn_lrelu = functools.partial(conv_norm_act, activation=lrelu, norm="InstanceNorm")

        self.dis = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                 conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                 conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                 # Self_Attn(dim * 4),
                                 conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),  # 1×512×31*37
                                 nn.Conv2d(dim * 8, 1, 4, 1, (2, 1))  # B×1×32*16
                                 )

    def forward(self, x, layers=[]):
        if len(layers) > 0:
            feats = []
            for layer_id, layer in enumerate(self.dis):
                x = layer(x)
                if layer_id in layers:
                    feats.append(x)
                    # print(layer)
            return feats
        else:
            return self.dis(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()

        conv_bn_relu = functools.partial(conv_norm_act, norm="InstanceNorm")
        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                    conv_bn_relu(in_dim, out_dim, 3, 1),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(out_dim, out_dim, 3, 1),
                                    nn.InstanceNorm2d(out_dim)
                                    )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):

    def __init__(self, dim=64):
        super(Generator, self).__init__()

        conv_bn_relu = functools.partial(conv_norm_act, norm="InstanceNorm")
        dconv_bn_relu = functools.partial(dconv_norm_act, norm="InstanceNorm")

        self.encoder = nn.Sequential(nn.ReflectionPad2d(3),
                                     conv_bn_relu(3, dim * 1, 7, 1),
                                     conv_bn_relu(dim * 1, dim * 2, 3, 2, 1),
                                     conv_bn_relu(dim * 2, dim * 4, 3, 2, 1),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     ResidualBlock(dim * 4, dim * 4),
                                     #  Self_Attn(dim * 4)
                                     )

        self.decoder = nn.Sequential(dconv_bn_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                                     dconv_bn_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                                     #  Self_Attn(dim * 1),
                                     nn.ReflectionPad2d(3),
                                     nn.Conv2d(dim, 3, 7, 1),
                                     nn.Tanh())

    def forward(self, x, layers=[], encode_only=False, feat_extractor=False):
        if feat_extractor:
            features = self.encoder(x)
            return features
        else:
            if len(layers) > 0:
                feat = x
                feats = []
                for layer_id, layer in enumerate(self.encoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1] and encode_only:
                        return feats  # return intermediate features alone; stop in the last layers
                feat = self.decoder(feat)
                return feat, feats  # return both output and intermediate results
            else:
                features = self.encoder(x)
                images = self.decoder(features)
                return images


class Conv_Relu_Pool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Relu_Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class Id_Net(nn.Module):
    def __init__(self, in_channels, n_identities):
        super(Id_Net, self).__init__()
        layers = [3, 4, 6, 3]
        # layers = [1, 1, 1, 1]
        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.inplanes = 256
        replace_stride_with_dilation = [False, False, False]
        self.layer2 = self._make_layer(
            Bottleneck,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2]
        )

        # self.layers = nn.Sequential(
        #     BasicBlock(in_channels, in_channels * 2, 2, conv1x1(in_channels, in_channels * 2, 2)),
        #     BasicBlock(in_channels * 2, in_channels * 4, 2, conv1x1(in_channels * 2, in_channels * 4, 2)),
        #     BasicBlock(in_channels * 4, in_channels * 8, 2, conv1x1(in_channels * 4, in_channels * 8, 2)),
        # )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bnneck = nn.BatchNorm1d(in_channels * 8)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(2048, n_identities)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        feats = self.layer4(x)

        # feats = self.layers(x)
        global_feats = self.global_avgpool(feats)
        global_feats = self.flatten(global_feats)
        global_feats_norm = self.bnneck(global_feats)
        if not self.training:
            return global_feats_norm
            # return global_feats
        out = self.classifier(global_feats_norm)
        # out = self.classifier(global_feats)
        return out, global_feats
