# from random import random
import torchreid
import torch.nn as nn
# import torch
from torchreid.utils import set_random_seed, load_pretrained_weights, max_ham_permutations
# from torchreid.utils import load_pretrained_weights
import sys
from torchsummary import summary

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
    batch_size_train=32,  # 14
    batch_size_test=100,
    # transforms=None,
    transforms=['random_flip', 'pad', 'random_crop'],
    val=False,
    #  combineall=True,
    # workers=4,
    norm_mean=[0.485, 0.456, 0.406],  # imagenet mean
    norm_std=[0.229, 0.224, 0.225],  # imagenet std
    load_train_targets=False,
    train_sampler='RandomIdentitySampler',
    num_instances=4
)


backbone_name = 'bnneck'

backbone = torchreid.models.build_model(
    name=backbone_name,
    num_classes=datamanager.num_train_pids,
    loss='triplet',
    # pretrained=False,
    last_stride=2
)

# backbone.cuda()
# print(backbone)
# sys.exit()

optimizer = torchreid.optim.build_optimizer(
    backbone_name,
    backbone,
    optim='adam',
    lr=1e-4,
    weight_decay=5e-4,
    # staged_lr=True,
    # new_layers=['classifier', 'layer4']
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1,
)

engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    backbone_name,
    backbone,
    optimizer=optimizer,
    scheduler=scheduler,
    val=False,
    generator_path='./log/CUT2/generator_S2R/model.pth.tar-8'
)

# load_pretrained_weights(backbone, 'log/prova_transfer1/model/model_source.pth.tar-30')
backbone = backbone.cuda()

engine.run(
    save_dir='log/prova_transfer6',
    max_epoch=30,
    eval_freq=6,
    print_freq=600,
    # open_layers=['classifier', 'layer4'],
    # fixbase_epoch=10
)

# engine.run(
#     save_dir='log/prova_transfer6',
#     test_only=True
# )
