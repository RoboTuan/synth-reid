# from random import random
import torchreid
import torch.nn as nn
import torchvision
from torchreid.models import self_sup
from torchreid.models import Generator, Discriminator
from torchreid.data.datasets import GTA_synthReid, Market1501
import torch
from PIL import Image
from torchreid.utils import set_random_seed, load_pretrained_weights, max_ham_permutations
from torchsummary import summary
import itertools
import os
import copy
import torchvision.transforms as transforms
from torch.utils.data.sampler import RandomSampler
import sys

import numpy as np
import random

set_random_seed()


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """`in_items` is a list of item."""
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


epochs = 6
batch_size = 3
lr = 0.0002
use_tensorboard = False
start_epoch = 0

# TODO:check the height and width that is different for each model (384x12, 256x128, ...)
datamanager = torchreid.data.ImageDataManager(
    root='/mnt/data2/defonte_data/PersonReid_datasets/',
    sources='gta_synthreid',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=3,  # 14
    batch_size_test=100,
    transforms=None,
    val=False,
    workers=0,
    load_train_targets=True
)

# load_size_w = 144
# load_size_h = 286
# crop_size_w = 128
# crop_size_h = 256
# transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(),
#      transforms.Resize((load_size_h, load_size_w), Image.BICUBIC),
#      transforms.RandomCrop((crop_size_h, crop_size_w)),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

# test_transform = transforms.Compose(
#     [transforms.Resize((crop_size_h, crop_size_w), Image.BICUBIC),
#      transforms.ToTensor(),
#      transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

# np.random.seed(10)
# random.seed(10)
# torch.manual_seed(10)
# gta_train = GTA_synthReid(root='/mnt/data2/defonte_data/PersonReid_datasets/', transform=transform, mode="train")

# np.random.seed(10)
# random.seed(10)
# torch.manual_seed(10)
# market_train = Market1501(root='/mnt/data2/defonte_data/PersonReid_datasets/', transform=transform, mode="train")

# a_loader = torch.utils.data.DataLoader(gta_train, batch_size=batch_size, sampler=RandomSampler(gta_train.train), num_workers=0)
# b_loader = torch.utils.data.DataLoader(market_train, batch_size=batch_size, sampler=RandomSampler(market_train.train), num_workers=0)

a_loader = datamanager.train_loader
b_loader = datamanager.train_loader_t

a_fake_pool = ItemPool()
b_fake_pool = ItemPool()


""" model """
Da = Discriminator()
Db = Discriminator()
Ga = Generator()
Gb = Generator()
MSE = nn.MSELoss()
L1 = nn.L1Loss()
cuda([Da, Db, Ga, Gb])

da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))

loss = {}
for epoch in range(start_epoch, epochs):
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # step
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # set train
        Ga.train()
        Gb.train()

        # leaves
        # a_real_pth = a_real['impath']
        # b_real_pth = b_real['impath']
        a_real = a_real['img']
        # print(a_real_pth)
        # a_real_test = (torch.unsqueeze(a_real[0], 0).data + 1) / 2.0
        # torchvision.utils.save_image(a_real_test, '%s/Epoch_(%d)_(%dof%d).jpg' % ("./", epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=1)

        # print(a_real.shape)
        b_real = b_real['img']
        # print(b_real.shape)
        # sys.exit()
        a_real, b_real = cuda([a_real, b_real])

        # train G
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)

        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        # gen losses
        a_f_dis = Da(a_fake)
        b_f_dis = Db(b_fake)
        r_label = cuda(torch.ones(a_f_dis.size()))
        a_gen_loss = MSE(a_f_dis, r_label)
        b_gen_loss = MSE(b_f_dis, r_label)

        # identity loss
        b2b = Gb(b_real)
        a2a = Ga(a_real)
        idt_loss_b = L1(b2b, b_real)
        idt_loss_a = L1(a2a, a_real)
        idt_loss = idt_loss_a + idt_loss_b
        # rec losses
        a_rec_loss = L1(a_rec, a_real)
        b_rec_loss = L1(b_rec, b_real)
        rec_loss = a_rec_loss + b_rec_loss
        # g loss
        g_loss = a_gen_loss + b_gen_loss + rec_loss * 10.0 + 5.0 * idt_loss
        loss['G/a_gen_loss'] = a_gen_loss.item()
        loss['G/b_gen_loss'] = b_gen_loss.item()
        loss['G/rec_loss'] = rec_loss.item()
        loss['G/idt_loss'] = idt_loss.item()
        loss['G/g_loss'] = g_loss.item()
        # backward
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss.backward()
        ga_optimizer.step()
        gb_optimizer.step()

        # leaves
        a_fake = torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0])
        b_fake = torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0])
        a_fake, b_fake = cuda([a_fake, b_fake])

        # train D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = cuda(torch.ones(a_f_dis.size()))
        f_label = cuda(torch.zeros(a_f_dis.size()))

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = (a_d_r_loss + a_d_f_loss) * 0.5
        b_d_loss = (b_d_r_loss + b_d_f_loss) * 0.5
        loss['D/a_d_f_loss'] = a_d_f_loss.item()
        loss['D/b_d_f_loss'] = b_d_f_loss.item()
        loss['D/a_d_r_loss'] = a_d_r_loss.item()
        loss['D/b_d_r_loss'] = b_d_r_loss.item()
        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, min((len(a_loader), len(b_loader)))))
            print("g_loss: (%f)  a_d_loss: (%f)   b_d_loss: (%f)" % (g_loss, a_d_loss, b_d_loss))
            if use_tensorboard:
                for tag, value in loss.items():
                    Logger.scalar_summary(tag, value, i)
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                Ga.eval()
                Gb.eval()
                a_real_test = torch.unsqueeze(a_real[0], 0)
                b_real_test = torch.unsqueeze(b_real[0], 0)
                # print(a_real_pth[0])
                # print(b_real_pth[0])

                # a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
                # train G
                a_fake_test = Ga(b_real_test)
                b_fake_test = Gb(a_real_test)

                a_rec_test = Ga(b_fake_test)
                b_rec_test = Gb(a_fake_test)

                pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0

                save_dir = './sample_images_while_training/cyclegan1'
                mkdir(save_dir)
                torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=3)
