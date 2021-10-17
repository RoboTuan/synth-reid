# from random import random
import torchreid
import torch.nn as nn
# import torch
from torchreid.utils import set_random_seed, load_pretrained_weights
# from torchreid.utils import load_pretrained_weights
import sys

import numpy as np
import random

set_random_seed()
# np.random.seed(10)
# random.seed(10)


# set_random_seed(0)
# from torchreid.data.datasets.image import GTA_synthReid
# #set_random_seed(0)
# gta = GTA_synthReid(val=True)
# print(gta.val_query[0])
# #set_random_seed(0)
# gta = GTA_synthReid(val=True)
# print(gta.val_query[0])
# #set_random_seed(0)
# gta = GTA_synthReid(val=True)
# print(gta.val_query[0])
# #set_random_seed(0)
# gta = GTA_synthReid(val=True)
# print(gta.val_query[0])
# #set_random_seed(0)
# gta = GTA_synthReid(val=True)
# print(gta.val_query[0])
# sys.exit()

datamanager = torchreid.data.ImageDataManager(
    root='/mnt/data2/defonte_data/PersonReid_datasets/',
    sources='gta_synthreid',
    targets='gta_synthreid',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    # transforms=['random_flip', 'random_crop'],
    transforms=['random_flip'],
    val=True
)
# print(torch.initial_seed())
# print(len(datamanager.train_loader))
# for data in datamanager.train_loader:
#     print(data['impath'][:5])
#     break

# print(len(datamanager.test_loader['gta_synthreid']['query']))
# for data in datamanager.test_loader['gta_synthreid']['query']:
#     print(data['impath'][:5])
#     break

# print(len(datamanager.test_loader['gta_synthreid']['gallery']))
# for data in datamanager.test_loader['gta_synthreid']['gallery']:
#     print(data['impath'][:5])
#     break

# print(len(datamanager.val_loader['gta_synthreid']['query']))
# for data in datamanager.val_loader['gta_synthreid']['query']:
#     print(data['impath'][:5])
#     break

# print(len(datamanager.val_loader['gta_synthreid']['gallery']))
# for data in datamanager.val_loader['gta_synthreid']['gallery']:
#     print(data['impath'][:5])
#     break

# sys.exit()

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)


print(model)
# model= nn.DataParallel(model)

# model.classifier = nn.Sequential()
# load_pretrained_weights(model, 'log/pcb_p6/model/model.pth.tar-10')
model = model.cuda()
# sys.exit()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='sgd',
    lr=0.05,
    sgd_nesterov=True,
    new_layers='classifier',
    staged_lr=True,
    base_lr_mult=0.1
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
    val=True
)

# sys.exit()

engine.run(
    save_dir='log/pcb_p6',
    max_epoch=1,
    eval_freq=4,
    print_freq=100,
    eval_flip=True
    # test_only=True
)
