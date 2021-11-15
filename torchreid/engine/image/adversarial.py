from __future__ import division, print_function, absolute_import

import torch.nn as nn
import torch

from torchreid import metrics
from torchreid.utils import ReplayBuffer
# from torchreid.losses import CrossEntropyLoss
# from torchreid.models.self_sup import SelfSup
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

        self.generated_S_buffer = ReplayBuffer()
        self.generated_R_buffer = ReplayBuffer()

        for name in model_names:
            print(name)
            self.register_model(name, self.models[name], self.optimizers[name], self.schedulers[name])

        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.adversarial_loss = torch.nn.MSELoss()

    def forward_backward_adversarial(self, data_S, data_R):
        imgs_S, _ = self.parse_data_for_train(data_S)
        imgs_R, _ = self.parse_data_for_train(data_R)
        batch_size = imgs_S.size(0)

        loss_summary = {}

        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32)

        if self.use_gpu:
            imgs_S = imgs_S.cuda()
            imgs_R = imgs_R.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        # generator, discriminator_S, discriminator_R
        ########################################
        # Update the two generators: S->R & R->S
        ########################################
        self.optimizers['generator'].zero_grad()

        # TODO: assign weights to the losses
        # Identity loss G_R2S(S) should be equal to S
        identity_S = self.models['generator'].generator_R2S(imgs_S)
        g_loss_identity_S = self.identity_loss(identity_S, imgs_S)
        g_loss_identity_S.backward()
        loss_summary['g_loss_identity_S'] = g_loss_identity_S
        # Identity loss G_S2R(R) should be equal to R
        identity_R = self.models['generator'].generator_S2R(imgs_R)
        g_loss_identity_R = self.identity_loss(identity_R, imgs_R)
        g_loss_identity_R.backward()
        loss_summary['g_loss_identity_R'] = g_loss_identity_R

        # Gan loss D_S(G_R2S(R))
        generated_imgs_S = self.models['generator'].generator_R2S(imgs_R)
        g_output_S = self.models['discriminator_S'](generated_imgs_S)
        g_loss_gan_R2S = self.adversarial_loss(g_output_S, real_label)
        g_loss_gan_R2S.backward(retain_graph=True)
        loss_summary['g_loss_gan_R2S'] = g_loss_gan_R2S
        # Gan loss D_R(G_S2R(S))
        generated_imgs_R = self.models['generator'].generator_S2R(imgs_S)
        g_output_R = self.models['discriminator_R'](generated_imgs_R)
        g_loss_gan_S2R = self.adversarial_loss(g_output_R, real_label)
        g_loss_gan_S2R.backward(retain_graph=True)
        loss_summary['g_loss_gan_S2R'] = g_loss_gan_S2R

        # Cycle loss G_R2S(G_S2R(S))
        recovered_imgs_S = self.models['generator'].generator_R2S(generated_imgs_R)
        g_loss_cycle_SRS = self.cycle_loss(recovered_imgs_S, imgs_S)
        g_loss_cycle_SRS.backward()
        loss_summary['g_loss_cycle_SRS'] = g_loss_cycle_SRS
        # Cycle loss G_S2R(G_R2S(R))
        recovered_imgs_R = self.models['generator'].generator_S2R(generated_imgs_S)
        g_loss_cycle_RSR = self.cycle_loss(recovered_imgs_R, imgs_R)
        g_loss_cycle_RSR.backward()
        loss_summary['g_loss_cycle_RSR'] = g_loss_cycle_RSR

        # Total generators loss
        # loss_generators = g_loss_identity_S + g_loss_identity_R +\
        #     g_loss_gan_R2S + g_loss_gan_S2R +\
        #     g_loss_cycle_SRS + g_loss_cycle_RSR
        # loss_generators.backwards()
        self.optimizers['generator'].step()

        ########################################
        # Update the synth discriminator: DS
        ########################################
        self.optimizers['discriminator_S'].zero_grad()

        # S image loss
        ds_output_S = self.models['discriminator_S'](imgs_S)
        ds_loss_S = self.adversarial_loss(ds_output_S, real_label)
        ds_loss_S.backward()
        loss_summary['ds_loss_S'] = ds_loss_S
        # G_R2S(R) image loss
        generated_imgs_S = self.generated_S_buffer.push_and_pop(generated_imgs_S)
        ds_output_R2S = self.models['discriminator_S'](generated_imgs_S.detach())
        ds_loss_R2S = self.adversarial_loss(ds_output_R2S, fake_label)
        ds_loss_R2S.backward()
        loss_summary['ds_loss_R2S'] = ds_loss_R2S

        # Total discriminator_S loss
        # loss_discriminator_S = ds_loss_S + ds_loss_R2S
        # loss_discriminator_S.backward()
        self.optimizers['discriminator_S'].step()

        ########################################
        # Update the real discriminator: DR
        ########################################
        self.optimizers['discriminator_R'].zero_grad()

        # R image loss
        dr_output_R = self.models['discriminator_R'](imgs_R)
        dr_loss_R = self.adversarial_loss(dr_output_R, real_label)
        dr_loss_R.backward()
        loss_summary['dr_loss_R'] = dr_loss_R
        # G_S2R(S) image loss
        generated_imgs_R = self.generated_R_buffer.push_and_pop(generated_imgs_R)
        dr_output_S2R = self.models['discriminator_R'](generated_imgs_R.detach())
        dr_loss_S2R = self.adversarial_loss(dr_output_S2R, fake_label)
        dr_loss_S2R.backward()
        loss_summary['dr_loss_S2R'] = dr_loss_S2R

        # Total discriminator_R loss
        # loss_discriminator_R = dr_loss_R + dr_loss_S2R
        # loss_discriminator_R.backward()
        self.optimizers['discriminator_R'].step()

        return loss_summary
