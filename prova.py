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
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=14,
    batch_size_test=100,
    transforms=['random_flip', 'pad', 'random_crop'],
    # transforms=['random_flip', 'pad', 'random_crop', 'random_erase'],
    val=False,
    combineall=True,
    load_train_targets=True
    # train_sampler='RandomIdentitySampler',
    # num_instances=4
)
# sys.exit()
generators_name = 'generator'
discriminator_S_name = 'discriminator_S'
discriminator_R_name = 'discriminator_R'

generator = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    name=generators_name,
    adversarial=True
)
discriminator_S = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    name=discriminator_S_name,
    adversarial=True
)
discriminator_R = torchreid.models.build_model(
    num_classes=datamanager.num_train_pids,
    name=discriminator_R_name,
    adversarial=True
)

generator.cuda()
discriminator_S.cuda()
discriminator_R.cuda()

# model = torchreid.models.build_model(
#     name=model_name,
#     num_classes=datamanager.num_train_pids,
#     loss='softmax',
#     # last_stride=2,
#     self_sup=True,
#     # neck='bnneck',
#     # neck_feat='after',
#     pretrained=True
# )
# print(model)
# for name, module in model.jig_saw_puzzle.named_children():
#     print(name)
#     for p in module.parameters():
#         print(p.requires_grad)
#     print()

# sys.exit()

# for data in datamanager.train_loader_t:
#     print(data)
#     break

# for data in datamanager.train_loader:
#     print(data)
#     # model(data['img'])
#     # print("Data")
#     # print(data['img'].shape)
#     break

# sys.exit()

# print(model)
# model = nn.DataParallel(model)

# model.classifier = nn.Sequential()
# load_pretrained_weights(model, 'log/pcb_p6/model/model.pth.tar-10')

# model = model.cuda()

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

optimizer_G = torchreid.optim.build_optimizer(
    generators_name,
    generator,
    optim='adam',
    lr=2e-4,
    # sgd_nesterov=True,
)

optimizer_DS = torchreid.optim.build_optimizer(
    discriminator_S_name,
    discriminator_S,
    optim='adam',
    lr=2e-4,
    # sgd_nesterov=True,
)

optimizer_DR = torchreid.optim.build_optimizer(
    discriminator_R_name,
    discriminator_R,
    optim='adam',
    lr=2e-4,
    # sgd_nesterov=True,
)

# scheduler = torchreid.optim.build_lr_scheduler(
#     optimizer,
#     lr_scheduler='warmup_multi_step',
#     stepsize=[15, 17],
#     warmup_iters=10,
#     warmup_factor=1 / 10,
#     warmup_method='linear',
# )

scheduler_G = torchreid.optim.build_lr_scheduler(
    optimizer_G,
    lr_scheduler='multi_step',
    stepsize=[15, 25],
    gamma=0.1
)

scheduler_DS = torchreid.optim.build_lr_scheduler(
    optimizer_DS,
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

model_names = [generators_name, discriminator_S_name, discriminator_R_name]
models = {
    'generator': generator,
    'discriminator_S': discriminator_S,
    'discriminator_R': discriminator_R,
}
optimizers = {
    'generator': optimizer_G,
    'discriminator_S': optimizer_DS,
    'discriminator_R': optimizer_DR,
}
schedulers = {
    'generator': scheduler_G,
    'discriminator_S': scheduler_DS,
    'discriminator_R': scheduler_DR,
}

engine = torchreid.engine.ImageAdversarialEngine(
    datamanager,
    model_names,
    models,
    optimizers=optimizers,
    schedulers=schedulers,
    val=False,
)

engine.run(
    save_dir='log/gan',
    max_epoch=4,
    eval_freq=6,
    print_freq=10,
)

# load_pretrained_weights(model, 'log/im_resnet50_softmax_val_open[3_4_cls]_multi/model/model.pth.tar-30')
# model = model.cuda()

# engine.run(
#     test_only=True
# )
