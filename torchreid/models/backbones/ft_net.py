from torch.nn import init
import torch.nn as nn
from .resnet import resnet50
from torchreid.utils import weights_init_kaiming, weights_init_classifier


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 droprate,
                 relu=False,
                 bnorm=True,
                 num_bottleneck=512,
                 linear=True,
                 return_f=False):

        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        # Return only the output of the second-last nn.Linear layer
        # during evaluation
        if not self.training:
            return x
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, num_classes, loss='softmax', pretrained=True, droprate=0.5, stride=2, circle=False, **kwargs):
        super(ft_net, self).__init__()
        model_ft = resnet50(pretrained=pretrained, loss=loss, num_classes=num_classes, **kwargs)
        # print(model_ft)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        # self.classifier = ClassBlock(2048, class_num,
        #                              droprate, return_f=circle)
        self.model.classifier = ClassBlock(2048, num_classes,
                                           droprate, return_f=circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.global_avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model.classifier(x)
        # x = self.classifier(x)
        return x


def ft_net50(num_classes, loss, pretrained=True, droprate=0.5, stride=2, circle=False, **kwargs):
    model = ft_net(num_classes, loss, pretrained, droprate, stride, circle, **kwargs)
    return model
