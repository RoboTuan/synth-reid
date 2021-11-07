# from random import random
import torchreid
import torch.nn as nn
from torchreid.models import self_sup
# import torch
from torchreid.utils import set_random_seed, load_pretrained_weights, max_ham_permutations
# from torchreid.utils import load_pretrained_weights
import sys
from torchsummary import summary

import numpy as np
import random

set_random_seed()
# np.random.seed(10)
# random.seed(10)

# max_p = 31
# P = max_ham_permutations(4, 2, max_p)
# np.save("./perm_" + str(max_p), P)
# sys.exit()


# TODO:check the height and width that is different for each model (384x12, 256x128, ...)
datamanager = torchreid.data.ImageDataManager(
    root='/mnt/data2/defonte_data/PersonReid_datasets/',
    sources='gta_synthreid',
    targets='gta_synthreid',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'pad', 'random_crop'],
    # transforms=['random_flip', 'pad', 'random_crop', 'random_erase'],
    val=True,
    # train_sampler='RandomIdentitySampler',
    # num_instances=4
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    # last_stride=2,
    self_sup=True,
    # neck='bnneck',
    # neck_feat='after',
    pretrained=True
)
# print(model)
# for name, module in model.jig_saw_puzzle.named_children():
#     print(name)
#     for p in module.parameters():
#         print(p.requires_grad)
#     print()

# sys.exit()

# for data in datamanager.train_loader:
#     model(data['img'])
#     print("Data")
#     print(data['img'].shape)
#     break

# sys.exit()

# print(model)
# model = nn.DataParallel(model)

# model.classifier = nn.Sequential()
# load_pretrained_weights(model, 'log/pcb_p6/model/model.pth.tar-10')
model = model.cuda()
# summary(model, (3, 256, 128))
# print(model)
# sys.exit()

new_layers_self_sup = [
    'backbone.classifier',
    # 'selfSup.feat_extractor',
    # 'selfSup.global_avgpool',
    # 'selfSup.flatten',
    # 'selfSup.classifier'
]

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='sgd',
    lr=3.5e-4,
    sgd_nesterov=True,
    new_layers=new_layers_self_sup,
    # new_layers=['classifier', 'layer4'],
    staged_lr=True,
    base_lr_mult=0.1,
    self_sup=True
)

# scheduler = torchreid.optim.build_lr_scheduler(
#     optimizer,
#     lr_scheduler='warmup_multi_step',
#     stepsize=[15, 17],
#     warmup_iters=10,
#     warmup_factor=1 / 10,
#     warmup_method='linear',
# )

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=False,
    val=True,
    self_sup=True,
)

open_layers_self_sup = [
    'backbone.classifier',
    'backbone.layer3',
    'backbone.layer4',
    'selfSup.feat_extractor',
    'selfSup.global_avgpool',
    'selfSup.flatten',
    'selfSup.classifier'
]

engine.run(
    save_dir='log/self_sup_resnet50_seed10_open_classifier',
    max_epoch=1,
    eval_freq=6,
    print_freq=400,
    eval_flip=True,
    fixbase_epoch=30,
    open_layers=open_layers_self_sup
    # test_only=True
)

load_pretrained_weights(model, 'log/self_sup_resnet50_seed10_open_classifier/model/model.pth.tar-30')
model = model.cuda()

engine.run(
    test_only=True
)
