from __future__ import division, print_function, absolute_import

import torch.nn as nn
import torch
import sys

from torchreid import metrics
from ..engine import Engine
from torchreid.losses import PatchNCELoss, CrossEntropyLoss, SimLoss, TripletLoss


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
        weight_nce=0.5,
        weight_idt=0.5,
        weight_gen=1.,
        weight_dis=0.5,
        weight_sim=0,
        weight_x=1.,
        weight_t=0.,
        epoch_id=1,
        sim_type_loss='feat_match',
        guide_gen=False,
        nce_layers=[0, 2, 3, 4, 8],
        dis_layers=[1, 2, 3, 4],
        num_patches=256,
    ):
        self.val = val
        self.adversarial = True

        super(ImageAdversarialEngine, self).__init__(datamanager=datamanager,
                                                     val=self.val,
                                                     use_gpu=use_gpu,
                                                     adversarial=self.adversarial)

        self.weight_nce = weight_nce
        self.weight_idt = weight_idt
        self.weight_gen = weight_gen
        self.weight_dis = weight_dis
        self.weight_sim = weight_sim
        self.weight_x = weight_x
        self.weight_t = weight_t
        self.guide_gen = guide_gen
        self.sim_type_loss = sim_type_loss
        self.epoch_id = epoch_id

        # optimizers = {
        #   generator: OPTg,
        #   disrcA: OPTa,
        #   discB: OPTb
        # }
        self.model_names = model_names
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.dis_layers = dis_layers
        batch_size = datamanager.batch_size_train

        for name in model_names.values():
            print(name)
            self.register_model(name, self.models[name], self.optimizers[name], self.schedulers[name])
            # if name != 'mlp':
            #     wandb.watch(self.models['id_net'].module, log='gradients', log_freq=250, log_graph=False)

        # self.optimizers[self.model_names['convnet']], self.schedulers[self.model_names['convnet']] =\
        #     self.models[self.model_names['convnet']].module.optim_sched()
        # self.register_model(self.model_names['convnet'],
        #                     self.models[self.model_names['convnet']],
        #                     self.optimizers[self.model_names['convnet']],
        #                     self.schedulers[self.model_names['convnet']])

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.criterionNCE = [PatchNCELoss(batch_size).cuda() for _ in self.nce_layers]
        self.simLoss = SimLoss(sim_type_loss)
        self.criterion_t = TripletLoss(margin=0.3)
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
        self.models[self.model_names['generator']].zero_grad()
        r_fake, r_fake_features = self.models[self.model_names['generator']](s_real, layers=[12])
        # gen losses
        if self.weight_sim > 0:
            dis_feats_fake = self.models[self.model_names['discriminator']](r_fake, self.dis_layers + [5])
            r_fake_gen = dis_feats_fake[-1]
        else:
            r_fake_gen = self.models[self.model_names['discriminator']](r_fake)
        real_label = torch.ones(r_fake_gen.size()).cuda()
        r_gen_loss = self.MSE(r_fake_gen, real_label) * self.weight_gen
        loss['r_gen_loss'] = r_gen_loss.item()
        g_loss = r_gen_loss

        # similarity losses
        if self.weight_sim > 0:
            dis_feats_real = self.models[self.model_names['discriminator']](r_real, self.dis_layers)
            gen_sim_loss = self.calculate_sim_loss(dis_feats_real, dis_feats_fake[:-1]) * self.weight_sim
            g_loss += gen_sim_loss
            loss['gen_sim_loss'] = gen_sim_loss.item()

        # nce loss
        if self.weight_nce > 0:
            nce_loss = self.calculate_NCE_loss(s_real, r_fake) * self.weight_gen
            loss['nce_loss'] = nce_loss.item()
            g_loss += nce_loss
            if self.optimizers[self.model_names['feature_net']] is None:
                print("Initializing the feature net on the fly")
                self.optimizers[self.model_names['feature_net']], self.schedulers[self.model_names['feature_net']] =\
                    self.models[self.model_names['feature_net']].module.optim_sched()
                self.register_model(self.model_names['feature_net'],
                                    self.models[self.model_names['feature_net']],
                                    self.optimizers[self.model_names['feature_net']],
                                    self.schedulers[self.model_names['feature_net']])
                # wandb.watch(self.models[self.model_names['feature_net']].module,
                #             log='gradients',
                #             log_freq=250,
                #             log_graph=False)
            self.models[self.model_names['feature_net']].zero_grad()

            # identity nce loss
            if self.weight_idt > 0:
                # compute identity loss only if the nce loss is considered
                r2r = self.models[self.model_names['generator']](r_real) * self.weight_idt
                idt_nce_loss = self.calculate_NCE_loss(r_real, r2r)
                loss['idt_nce_loss'] = idt_nce_loss.item()
                g_loss += idt_nce_loss

        if self.guide_gen is True and self.weight_x > 0 and (self.epoch + 1) >= self.epoch_id:
            self.models[self.model_names['convnet']].zero_grad()
            # Re-Id loss guiding the generator
            r_fake_outputs, trip_feats = self.models[self.model_names['convnet']](r_fake_features[0])
            reid_loss = self.compute_loss(self.criterion, r_fake_outputs, s_pids) * self.weight_x
            loss['reid_loss'] = reid_loss.item()
            g_loss += reid_loss
            loss['acc'] = metrics.accuracy(r_fake_outputs, s_pids)[0].item()
            if self.weight_t > 0:
                t_reid_loss = self.compute_loss(self.criterion_t, trip_feats, s_pids) * self.weight_t
                loss['t_reid_loss'] = t_reid_loss.item()
                g_loss += t_reid_loss

        # loss
        g_loss.backward()
        self.optimizers[self.model_names['generator']].step()
        if self.weight_nce > 0:
            self.optimizers[self.model_names['feature_net']].step()
        if self.guide_gen is True and self.weight_x > 0:
            self.optimizers[self.model_names['convnet']].step()

        # leaves
        # r_fake = self.r_fake_pool.push_and_pop(r_fake.detach().cpu())
        # r_fake = r_fake.cuda()
        r_fake = r_fake.detach()
        s_real = s_real.detach()

        # Discriminators
        # train D
        self.models[self.model_names['discriminator']].zero_grad()
        r_real_dis = self.models[self.model_names['discriminator']](r_real)
        if self.weight_sim > 0:
            dis_feats_fake = self.models[self.model_names['discriminator']](r_fake, self.dis_layers + [5])
            r_fake_dis = dis_feats_fake[-1]
        else:
            r_fake_dis = self.models[self.model_names['discriminator']](r_fake)
        real_label = torch.ones(r_real_dis.size()).cuda()
        fake_label = torch.zeros(r_real_dis.size()).cuda()

        # d loss
        r_dis_real_loss = self.MSE(r_real_dis, real_label) * self.weight_dis
        loss['r_dis_real_loss'] = r_dis_real_loss.item()
        dis_loss = r_dis_real_loss
        r_dis_fake_loss = self.MSE(r_fake_dis, fake_label) * self.weight_dis
        loss['r_dis_fake_loss'] = r_dis_fake_loss.item()
        dis_loss += r_dis_fake_loss

        # similarity losses
        if self.weight_sim > 0:
            dis_feats_synth = self.models[self.model_names['discriminator']](s_real, self.dis_layers)
            dis_sim_loss = self.calculate_sim_loss(dis_feats_synth, dis_feats_fake[:-1]) * self.weight_sim
            loss['dis_sim_loss'] = dis_sim_loss.item()
            dis_loss += dis_sim_loss

        # backward
        dis_loss.backward()
        self.optimizers[self.model_names['discriminator']].step()

        if not self.guide_gen and self.weight_x > 0 and (self.epoch + 1) >= self.epoch_id:
            # Re-Id loss independent of the generator
            self.models[self.model_names['convnet']].zero_grad()
            r_fake_features = r_fake_features[0].detach()
            # r_fake_features = self.models[self.model_names['generator']](r_fake, feat_extractor=True)
            r_fake_outputs, trip_feats = self.models[self.model_names['convnet']](r_fake_features)
            reid_loss = self.compute_loss(self.criterion, r_fake_outputs, s_pids) * self.weight_x
            loss['reid_loss'] = reid_loss.item()
            loss['acc'] = metrics.accuracy(r_fake_outputs, s_pids)[0].item()
            if self.weight_t > 0:
                t_reid_loss = self.compute_loss(self.criterion_t, trip_feats, s_pids) * self.weight_t
                loss['t_reid_loss'] = t_reid_loss.item()
                reid_loss += t_reid_loss
            reid_loss.backward()
            self.optimizers[self.model_names['convnet']].step()

        if self.weight_idt > 0:
            return loss, s_real, r_real, r2r
        else:
            return loss, s_real, r_real, r_real

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.models[self.model_names['generator']](tgt, self.nce_layers, encode_only=True)

        feat_k = self.models[self.model_names['generator']](src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.models[self.model_names['feature_net']](feat_k, self.num_patches, None)
        feat_q_pool, _ = self.models[self.model_names['feature_net']](feat_q, self.num_patches, sample_ids)

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
