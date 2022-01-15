# from random import random
import torchreid
import torch.nn as nn
from torchreid.models import self_sup
import torch
from itertools import repeat
from torchreid.utils import set_random_seed, load_pretrained_weights, resume_from_checkpoint
# from torchreid.utils import load_pretrained_weights
import sys
from torchsummary import summary
from collections import defaultdict, OrderedDict

import numpy as np
import random

set_random_seed()

# TODO:check the height and width that is different for each model (384x12, 256x128, ...)
datamanager = torchreid.data.ImageDataManager(
    root='/mnt/data2/defonte_data/PersonReid_datasets/',
    sources='gta_synthreid',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=8,  # 14
    batch_size_test=100,
    # transforms=None,
    transforms=['random_flip', 'random_crop_translate'],
    val=False,
    combineall=True,
    # workers=4,
    load_train_targets=True,
    norm_mean=[0.5] * 3,
    norm_std=[0.5] * 3,
    # train_sampler='RandomDatasetSampler',
    # num_instances=2
    n_samples=20  # taking only at max 20 images per identity for GTA_synthReid
)
# sys.exit()

# backbone_name = 'metric_net'
generator_S2R_name = 'generator_S2R'
# generator_R2S_name = 'generator_R2S'
# discriminator_S_name = 'discriminator_S'
discriminator_R_name = 'discriminator_R'
mlp_name = 'mlp'
id_net_name = 'id_net'

# backbone = torchreid.models.build_model(
#     num_classes=datamanager.num_train_pids,
#     name=backbone_name,
#     adversarial=True,
#     loss='triplet'
# )

generator_S2R = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    name=generator_S2R_name,
    adversarial=True
)

# print(generator_S2R)
# sys.exit()

# generator_R2S = torchreid.models.build_model(
#     num_classes=datamanager.num_train_pids,
#     name=generator_R2S_name,
#     adversarial=True
# )

# discriminator_S = torchreid.models.build_model(
#     num_classes=datamanager.num_train_pids,
#     name=discriminator_S_name,
#     adversarial=True
# )

discriminator_R = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    name=discriminator_R_name,
    adversarial=True
)

mlp = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    adversarial=True,
    name=mlp_name,
    use_mlp=True,
    nc=256
)

print(datamanager.num_train_pids)
id_net = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    in_planes=256,
    name=id_net_name,
    adversarial=True,
)

# generator_S2R.cuda()
# generator_R2S.cuda()
# discriminator_S.cuda()
# discriminator_R.cuda()

# backbone = nn.DataParallel(backbone).cuda()
generator_S2R = nn.DataParallel(generator_S2R).cuda()
# generator_R2S = nn.DataParallel(generator_R2S).cuda()
# discriminator_S = nn.DataParallel(discriminator_S).cuda()
discriminator_R = nn.DataParallel(discriminator_R).cuda()
mlp = nn.DataParallel(mlp).cuda()
id_net = nn.DataParallel(id_net).cuda()

# optimizer_B = torchreid.optim.build_optimizer(
#     backbone_name,
#     backbone,
#     optim='adam',
#     lr=2e-4,
#     # weight_decay=5e-2
#     # sgd_nesterov=True,
# )

optimizer_GS2R = torchreid.optim.build_optimizer(
    generator_S2R_name,
    generator_S2R,
    optim='adam',
    adam_beta1=0.5,
    lr=2e-4,
    # sgd_nesterov=True,
)

# optimizer_GR2S = torchreid.optim.build_optimizer(
#     generator_R2S_name,
#     generator_R2S,
#     optim='adam',
#     adam_beta1=0.5,
#     lr=2e-4,
#     # sgd_nesterov=True,
# )

# optimizer_DS = torchreid.optim.build_optimizer(
#     discriminator_S_name,
#     discriminator_S,
#     optim='adam',
#     adam_beta1=0.5,
#     lr=2e-4,
#     # sgd_nesterov=True,
# )

optimizer_DR = torchreid.optim.build_optimizer(
    discriminator_R_name,
    discriminator_R,
    optim='adam',
    adam_beta1=0.5,
    lr=2e-4,
    # sgd_nesterov=True,
)

optimizer_id = torchreid.optim.build_optimizer(
    id_net_name,
    id_net,
    optim='adam',
    adam_beta1=0.5,
    lr=2e-4,
)


# scheduler_B = torchreid.optim.build_lr_scheduler(
#     optimizer_B,
#     lr_scheduler='multi_step',
#     stepsize=[15, 25],
#     gamma=0.1
# )

scheduler_GS2R = torchreid.optim.build_lr_scheduler(
    optimizer_GS2R,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

# scheduler_GR2S = torchreid.optim.build_lr_scheduler(
#     optimizer_GR2S,
#     lr_scheduler='multi_step',
#     stepsize=[15, 25],
#     gamma=0.1
# )

# scheduler_DS = torchreid.optim.build_lr_scheduler(
#     optimizer_DS,
#     lr_scheduler='multi_step',
#     stepsize=[15, 25],
#     gamma=0.1
# )

scheduler_DR = torchreid.optim.build_lr_scheduler(
    optimizer_DR,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

scheduler_id = torchreid.optim.build_lr_scheduler(
    optimizer_id,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)


# model_names = [generator_S2R_name, generator_R2S_name, discriminator_S_name, discriminator_R_name]
model_names = [id_net_name, generator_S2R_name, discriminator_R_name, mlp_name]

models = {
    # 'metric_net': backbone,
    'generator_S2R': generator_S2R,
    # 'generator_R2S': generator_R2S,
    # 'discriminator_S': discriminator_S,
    'discriminator_R': discriminator_R,
    'mlp': mlp,
    'id_net': id_net
}

optimizers = {
    # 'metric_net': optimizer_B,
    'generator_S2R': optimizer_GS2R,
    # 'generator_R2S': optimizer_GR2S,
    # 'discriminator_S': optimizer_DS,
    'discriminator_R': optimizer_DR,
    'mlp': None,
    'id_net': optimizer_id
}

schedulers = {
    # 'metric_net': scheduler_B,
    'generator_S2R': scheduler_GS2R,
    # 'generator_R2S': scheduler_GR2S,
    # 'discriminator_S': scheduler_DS,
    'discriminator_R': scheduler_DR,
    'mlp': None,
    'id_net': scheduler_id
}

# for name in model_names:
#     if name == 'mlp':
#         dummy_feats = [torch.rand((1, 3)), torch.rand((1, 128))]
#         dummy_feats.extend([torch.rand((1, 256))] * 3)
#         mlp.module.create_mlp(dummy_feats)
#         optimizer_mlp, scheduler_mlp = mlp.module.optim_sched()
#         optimizers['mlp'] = optimizer_mlp
#         schedulers['mlp'] = scheduler_mlp
#     path = "log/CUT7/" + name + "/model.pth.tar-30"
#     start_epoch = resume_from_checkpoint(path, models[name], optimizers[name], schedulers[name])
#     print(start_epoch)

engine = torchreid.engine.ImageAdversarialEngine(
    datamanager,
    model_names,
    models,
    optimizers=optimizers,
    schedulers=schedulers,
    val=False,
)

engine.run(
    # start_epoch=start_epoch,
    save_dir='log/CUT9',
    max_epoch=8,
    eval_freq=6,
    print_freq=250,
)

# load_pretrained_weights(model, 'log/im_resnet50_softmax_val_open[3_4_cls]_multi/model/model.pth.tar-30')
# model = model.cuda()

# engine.run(
#     test_only=True,
#     rerank=True
# )
