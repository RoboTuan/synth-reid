from __future__ import absolute_import
# from torchreid.models.backbones.ft_net import ft_net

from .backbones import *
from .self_sup import SelfSup
from .adversarial_model import make_discriminator, make_generator, make_id_net
#from .models import *

__model_factory = {
    # image classification models
    'generator': Generator,
    'mlp': MLP,
    'id_net': Id_Net,
    # Model names must be unique when registering model,
    # so I added the 2 different discriminator of the same class
    'discriminator': Discriminator,
    'ft_net50' : ft_net50,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'bnneck': BNNeck,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_ain_x0_75': osnet_ain_x0_75,
    'osnet_ain_x0_5': osnet_ain_x0_5,
    'osnet_ain_x0_25': osnet_ain_x0_25
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


# def build_model(
#     name, num_classes, loss='softmax', pretrained=True, use_gpu=True, ft_net=False
# ):
def build_model(
    name='model', num_classes=None, loss='softmax', pretrained=True, use_gpu=True, self_sup=False, adversarial=False, **kwargs
):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    
    if not adversarial:
        # if ft_net :
        #     return models.ft_net(num_classes)
        backbone = __model_factory[name](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **kwargs
        )

        if self_sup:
            model = SelfSup(backbone, 2, 4, **kwargs)
        else:
            model = backbone

        return model
    else:
        # 'S' means synthetic, 'R' means real
        # S2R means transfer learning from synth to real
        if name == 'generator':
            return make_generator()
            # model = __model_factory[name]()
        elif name == 'discriminator':
            return make_discriminator()
        elif name == 'mlp':
            # return make_mlp(kwargs.get('use_mlp'), kwargs.get('nc'), use_gpu=True)
            return MLP(use_gpu=use_gpu, **kwargs)
        elif name == 'id_net':
            return make_id_net(kwargs.get('in_planes'), num_classes)
        else:
            backbone = __model_factory[name](
                # num_classes=num_classes,
                # loss=loss,
                # pretrained=pretrained,
                # use_gpu=use_gpu,
                # **kwargs
            )
            return backbone

