# from random import random
import torchreid
import torch.nn as nn
import torch
from itertools import repeat
from torchreid.utils import set_random_seed, load_pretrained_weights, resume_from_checkpoint
# from torchreid.utils import load_pretrained_weights
import sys
from torchsummary import summary
import wandb
import json

import numpy as np
import random

seed = 10
set_random_seed(seed)

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
    adversarial=True,
    combineall=True,
    seed=seed,
    # workers=4,
    load_train_targets=True,
    norm_mean=[0.5] * 3,
    norm_std=[0.5] * 3,
    train_sampler='RandomIdentitySampler',
    num_instances=4,
    n_samples=20  # taking only at max 20 images per identity for GTA_synthReid
)
# sys.exit()

generator_S2R_name = 'generator'
discriminator_R_name = 'discriminator'
feature_net_name = 'mlp'
convnet_name = 'id_net'


generator_S2R = torchreid.models.build_model(
    name=generator_S2R_name,
    adversarial=True
)

discriminator_R = torchreid.models.build_model(
    name=discriminator_R_name,
    adversarial=True
)

feature_net = torchreid.models.build_model(
    adversarial=True,
    name=feature_net_name,
    use_mlp=True,
    nc=256
)

print(datamanager.num_train_pids)
convnet = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    in_planes=256,
    name=convnet_name,
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
feature_net = nn.DataParallel(feature_net).cuda()
convnet = nn.DataParallel(convnet).cuda()


optimizer_GS2R = torchreid.optim.build_optimizer(
    generator_S2R_name,
    generator_S2R,
    optim='adam',
    adam_beta1=0.5,
    lr=2e-4,
    # sgd_nesterov=True,
)

optimizer_DR = torchreid.optim.build_optimizer(
    discriminator_R_name,
    discriminator_R,
    optim='adam',
    adam_beta1=0.5,
    lr=2e-4,
    # sgd_nesterov=True,
)

optimizer_CNN = torchreid.optim.build_optimizer(
    convnet_name,
    convnet,
    optim='adam',
    adam_beta1=0.5,
    lr=1e-5,
    weight_decay=5e-2
)


scheduler_GS2R = torchreid.optim.build_lr_scheduler(
    optimizer_GS2R,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

scheduler_DR = torchreid.optim.build_lr_scheduler(
    optimizer_DR,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

scheduler_CNN = torchreid.optim.build_lr_scheduler(
    optimizer_CNN,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1,
    # warmup_iters=10,
    # warmup_method='linear',
)


# model_names = [generator_S2R_name, generator_R2S_name, discriminator_S_name, discriminator_R_name]
# [convnet_name, generator_S2R_name, discriminator_R_name, feature_net_name]
model_names = {
    'generator': generator_S2R_name,
    'discriminator': discriminator_R_name,
    'feature_net': feature_net_name,
    'convnet': convnet_name
}

models = {
    generator_S2R_name: generator_S2R,
    discriminator_R_name: discriminator_R,
    feature_net_name: feature_net,
    convnet_name: convnet
}

optimizers = {
    generator_S2R_name: optimizer_GS2R,
    discriminator_R_name: optimizer_DR,
    feature_net_name: None,
    convnet_name: optimizer_CNN
}

schedulers = {
    generator_S2R_name: scheduler_GS2R,
    discriminator_R_name: scheduler_DR,
    feature_net_name: None,
    convnet_name: scheduler_CNN
}

# for name in model_names:
#     if name == 'mlp':
#         dummy_feats = [torch.rand((1, 3)), torch.rand((1, 128))]
#         dummy_feats.extend([torch.rand((1, 256))] * 3)
#         feature_net.module.create_mlp(dummy_feats)
#         optimizer_mlp, scheduler_mlp = feature_net.module.optim_sched()
#         optimizers['mlp'] = optimizer_mlp
#         schedulers['mlp'] = scheduler_mlp
#     path = "log/CUT10/" + name + "/model.pth.tar-30"
#     start_epoch = resume_from_checkpoint(path, models[name], optimizers[name], schedulers[name])
#     print(start_epoch)

try:
    with open('wandb_adv_api.key', 'r') as file:
        wandb_identity = json.load(file)
except FileNotFoundError as e:
    print("Create a json file in the root directory of this repo called 'wandb_api.key',\n")
    print("it must have as keys the 'entity', 'project' and 'api-key' of your wandb account.\n")
    print("For the entity you can put yours or that of a team,\
    for the project put a project name (e.g. adversarial_ReId")
    raise(e)
wandb.login(anonymous='never', relogin=True, timeout=30, key=wandb_identity['key'])
wandb.init(resume=False,
           # sync_tensorboard=True,
           mode="disabled",
           project=wandb_identity['project'],
           entity=wandb_identity['entity'],
           name="prova1")

engine = torchreid.engine.ImageAdversarialEngine(
    datamanager,
    model_names,
    models,
    optimizers=optimizers,
    schedulers=schedulers,
    val=False,
    weight_idt=0.,
    weight_t=1.,
    epoch_id=2
)

# open_layers = {
#     generator_S2R_name: None,
#     discriminator_R_name: None,
#     feature_net_name: None,
#     convnet_name: []
# }

engine.run(
    # start_epoch=start_epoch,
    save_dir='log/prova1',
    max_epoch=4,
    eval_freq=1,
    print_freq=500,
    # fixbase_epoch=10,
    # open_layers=open_layers
)

# load_pretrained_weights(model, 'log/im_resnet50_softmax_val_open[3_4_cls]_multi/model/model.pth.tar-30')
# model = model.cuda()

# engine.run(
#     test_only=True,
#     rerank=True
# )
