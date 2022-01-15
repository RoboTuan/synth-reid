from __future__ import division, print_function, absolute_import

import torch.nn as nn
import torch
import sys

from torchreid.utils import ReplayBuffer
from ..engine import Engine


class ImageAdversarialEngine(Engine):
    r"""Adversarial engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model names (List(String)): model names
        models (Dict(nn.Module)): model instance.
        optimizers (Dict(Optimizer)): The optimizers for generator, discriminator.
        schedulers (Dict(LRScheduler), optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        val (bool, optional): set to True if a validation set is used from part of the training set.

    """

    def __init__(
        self,
        datamanager,
        model_names,
        models,
        optimizers,
        schedulers=None,
        use_gpu=True,
        val=False,
        self_sup=False,
        lambda_id=1,
        lambda_ss=1,
    ):
        self.val = val
        self.self_sup = self_sup
        self.lambda_id = lambda_id
        self.lambda_ss = lambda_ss
        self.adversarial = True

        super(ImageAdversarialEngine, self).__init__(datamanager=datamanager,
                                                     val=self.val,
                                                     self_sup=self.self_sup,
                                                     lambda_id=self.lambda_id,
                                                     lambda_ss=self.lambda_ss,
                                                     use_gpu=use_gpu,
                                                     adversarial=self.adversarial)
        # optimizers = {
        #   generator: OPTg,
        #   disrcA: OPTa,
        #   discB: OPTb
        # }
        self.model_names = model_names
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.s_fake_pool = ReplayBuffer()
        self.r_fake_pool = ReplayBuffer()

        for name in model_names:
            print(name)
            self.register_model(name, self.models[name], self.optimizers[name], self.schedulers[name])

        # self.cycle_loss = nn.L1Loss()
        self.L1 = nn.L1Loss()
        self.MSE = torch.nn.MSELoss()

    def forward_backward_adversarial(self, batch_idx, data_S, data_R):
        # a==S  b==R
        # s_real means a "real" image from the "s"ynthetic domain, r_real means a "real" image from the "r"eal domain
        s_real, _ = self.parse_data_for_train(data_S)
        r_real, _ = self.parse_data_for_train(data_R)
        s_real, r_real = s_real.cuda(), r_real.cuda()
        loss = {}

        # Generators
        # train G
        s_fake = self.models['generator_R2S'](r_real)
        r_fake = self.models['generator_S2R'](s_real)

        # gen losses
        s_fake_gen = self.models['discriminator_S'](s_fake)
        r_fake_gen = self.models['discriminator_R'](r_fake)
        real_label = torch.ones(s_fake_gen.size()).cuda()
        s_gen_loss = self.MSE(s_fake_gen, real_label)
        r_gen_loss = self.MSE(r_fake_gen, real_label)
        gen_loss = s_gen_loss + r_gen_loss
        loss['gen_loss'] = gen_loss.item()

        # identity loss
        r2r = self.models['generator_S2R'](r_real)
        s2s = self.models['generator_R2S'](s_real)
        idt_loss_r = self.L1(r2r, r_real)
        idt_loss_s = self.L1(s2s, s_real)
        idt_loss = idt_loss_s + idt_loss_r
        loss['idt_loss'] = idt_loss.item()

        # rec losses
        s_rec = self.models['generator_R2S'](r_fake)
        r_rec = self.models['generator_S2R'](s_fake)
        s_rec_loss = self.L1(s_rec, s_real)
        r_rec_loss = self.L1(r_rec, r_real)
        rec_loss = s_rec_loss + r_rec_loss
        loss['rec_loss'] = rec_loss.item()

        # total g loss
        g_loss = gen_loss + rec_loss * 10.0 + 5.0 * idt_loss
        self.models['generator_R2S'].zero_grad()
        self.models['generator_S2R'].zero_grad()
        g_loss.backward()
        self.optimizers['generator_R2S'].step()
        self.optimizers['generator_S2R'].step()

        # leaves
        s_fake = self.s_fake_pool.push_and_pop(s_fake.cpu())
        r_fake = self.r_fake_pool.push_and_pop(r_fake.cpu())
        s_fake, r_fake = s_fake.cuda(), r_fake.cuda()

        # Discriminators
        # train D
        s_real_dis = self.models['discriminator_S'](s_real)
        s_fake_dis = self.models['discriminator_S'](s_fake)
        r_real_dis = self.models['discriminator_R'](r_real)
        r_fake_dis = self.models['discriminator_R'](r_fake)
        real_label = torch.ones(s_fake_dis.size()).cuda()
        fake_label = torch.zeros(s_fake_dis.size()).cuda()

        # d loss
        s_dis_real_loss = self.MSE(s_real_dis, real_label)
        s_dis_fake_loss = self.MSE(s_fake_dis, fake_label)
        r_dis_real_loss = self.MSE(r_real_dis, real_label)
        r_dis_fake_loss = self.MSE(r_fake_dis, fake_label)

        s_dis_loss = (s_dis_real_loss + s_dis_fake_loss) * 0.5
        r_dis_loss = (r_dis_real_loss + r_dis_fake_loss) * 0.5
        loss['s_dis_loss'] = s_dis_loss.item()
        loss['r_dis_loss'] = r_dis_loss.item()

        # backward
        self.models['discriminator_S'].zero_grad()
        self.models['discriminator_R'].zero_grad()
        s_dis_loss.backward()
        r_dis_loss.backward()
        self.optimizers['discriminator_S'].step()
        self.optimizers['discriminator_R'].step()

        return loss, s_real, r_real
