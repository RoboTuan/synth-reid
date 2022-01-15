from __future__ import division, print_function, absolute_import

import torch.nn as nn
import torch
import sys

from torchreid import metrics
from torchreid.utils import ReplayBuffer
from ..engine import Engine
from torchreid.losses import PatchNCELoss, CrossEntropyLoss, SimLoss
from torchvision.transforms import RandomPerspective, RandomHorizontalFlip


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
        nce_layers=[0, 2, 3, 4, 8],
        dis_layers=[1, 2, 3, 4],
        num_pathces=256,
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
        self.nce_layers = nce_layers
        self.num_patches = num_pathces
        self.dis_layers = dis_layers
        batch_size = datamanager.batch_size_train
        self.augmentations = [RandomHorizontalFlip(p=1),
                              RandomPerspective(distortion_scale=0.6, p=1.0)]
        # print(batch_size)

        for name in model_names:
            print(name)
            self.register_model(name, self.models[name], self.optimizers[name], self.schedulers[name])

        # self.optimizers['id_net'], self.schedulers['id_net'] =\
        #     self.models['id_net'].module.optim_sched()
        # self.register_model('id_net', self.models['id_net'], self.optimizers['id_net'], self.schedulers['id_net'])

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.criterionNCE = [PatchNCELoss(batch_size).cuda() for _ in self.nce_layers]
        self.simLoss = SimLoss()
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=True
        )

    def forward_backward_adversarial(self, batch_idx, data_S, data_R):
        s_real, s_pids = self.parse_data_for_train(data_S)
        r_real, _ = self.parse_data_for_train(data_R)
        s_real, r_real, s_pids = s_real.cuda(), r_real.cuda(), s_pids.cuda()
        loss = {}

        # Generators
        # train G
        r_fake = self.models['generator_S2R'](s_real)

        # gen losses
        # r_fake_gen = self.models['discriminator_R'](r_fake)
        dis_feats_fake = self.models['discriminator_R'](r_fake, self.dis_layers + [5])
        r_fake_gen = dis_feats_fake[-1]
        real_label = torch.ones(r_fake_gen.size()).cuda()
        r_gen_loss = self.MSE(r_fake_gen, real_label)
        loss['r_gen_loss'] = r_gen_loss.item()

        # similarity losses
        dis_feats_real = self.models['discriminator_R'](r_real, self.dis_layers)
        gen_sim_loss = self.calculate_sim_loss(dis_feats_real, dis_feats_fake[:-1])
        loss['gen_sim_loss'] = gen_sim_loss.item()

        # nce loss
        loss_nce = self.calculate_NCE_loss(s_real, r_fake) * 0.5
        loss['loss_nce'] = loss_nce.item()
        if self.optimizers['mlp'] is None:
            self.optimizers['mlp'], self.schedulers['mlp'] =\
                self.models['mlp'].module.optim_sched()
            self.register_model('mlp', self.models['mlp'], self.optimizers['mlp'], self.schedulers['mlp'])

        # identity nce loss
        r2r = self.models['generator_S2R'](r_real) * 0.5
        loss_idt_nce = self.calculate_NCE_loss(r_real, r2r)
        loss['loss_idt_nce'] = loss_idt_nce.item()

        # loss
        g_loss = r_gen_loss + loss_nce + loss_idt_nce + gen_sim_loss
        # g_loss = r_gen_loss + loss_nce + loss_idt_nce
        self.models['mlp'].zero_grad()
        self.models['generator_S2R'].zero_grad()
        g_loss.backward()
        self.optimizers['mlp'].step()
        self.optimizers['generator_S2R'].step()

        # leaves
        # r_fake = self.r_fake_pool.push_and_pop(r_fake.detach().cpu())
        # r_fake = r_fake.cuda()
        r_fake = r_fake.detach()
        s_real = s_real.detach()

        # Discriminators
        # train D
        r_real_dis = self.models['discriminator_R'](r_real)
        # r_fake_dis = self.models['discriminator_R'](r_fake)
        dis_feats_fake = self.models['discriminator_R'](r_fake, self.dis_layers + [5])
        r_fake_dis = dis_feats_fake[-1]
        real_label = torch.ones(r_real_dis.size()).cuda()
        fake_label = torch.zeros(r_real_dis.size()).cuda()

        # similarity losses
        dis_feats_synth = self.models['discriminator_R'](s_real, self.dis_layers)
        dis_sim_loss = self.calculate_sim_loss(dis_feats_synth, dis_feats_fake[:-1])
        loss['dis_sim_loss'] = dis_sim_loss.item()

        # d loss
        r_dis_real_loss = self.MSE(r_real_dis, real_label) * 0.5
        r_dis_fake_loss = self.MSE(r_fake_dis, fake_label) * 0.5

        r_dis_loss = (r_dis_real_loss + r_dis_fake_loss) + dis_sim_loss
        # r_dis_loss = (r_dis_real_loss + r_dis_fake_loss)
        loss['r_dis_real_loss'] = r_dis_real_loss.item()
        loss['r_dis_fake_loss'] = r_dis_fake_loss.item()

        # backward
        self.models['discriminator_R'].zero_grad()
        r_dis_loss.backward()
        self.optimizers['discriminator_R'].step()

        # Re-Id loss
        r_fake = r_fake.detach()
        r_fake_features = self.models['generator_S2R'](r_fake, feat_extractor=True)
        r_fake_outputs = self.models['id_net'](r_fake_features)
        reid_loss = self.compute_loss(self.criterion, r_fake_outputs, s_pids)
        loss['reid_loss'] = reid_loss.item()
        loss['acc'] = metrics.accuracy(r_fake_outputs, s_pids)[0].item()
        self.models['id_net'].zero_grad()
        reid_loss.backward()
        self.optimizers['id_net'].step()

        return loss, s_real, r_real, r2r

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.models['generator_S2R'](tgt, self.nce_layers, encode_only=True)

        feat_k = self.models['generator_S2R'](src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.models['mlp'](feat_k, self.num_patches, None)
        feat_q_pool, _ = self.models['mlp'](feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_sim_loss(self, feats_real, feats_fake):
        n_feats = len(feats_real)
        total_sim_loss = 0.0
        for feat_r, feat_f in zip(feats_real, feats_fake):
            loss = self.simLoss(feat_r, feat_f)
            total_sim_loss += loss
        return total_sim_loss / n_feats
