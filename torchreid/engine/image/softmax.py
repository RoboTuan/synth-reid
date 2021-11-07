from __future__ import division, print_function, absolute_import

import torch.nn as nn

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss
from torchreid.models.self_sup import SelfSup

from ..engine import Engine


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        val (bool, optional): set to True if a validation set is used from part of the training set.
        self_sup (bool, optional): set to True if the 'jig-saw puzzle' self supervised task is used.
        lambda_id (float, optional): if self_sup is True, this parameter controls the weight of the
            classification loss.
        lambda_ss (float, optional): if self_sup is True, this parameter controls the weight of the
            self supervised loss.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        val=False,
        self_sup=False,
        lambda_id=1,
        lambda_ss=1
    ):
        self.val = val
        self.self_sup = self_sup
        self.lambda_id = lambda_id
        self.lambda_ss = lambda_ss

        # if alpha > 1 or alpha < 0:
        #     raise ValueError("The value of 'alpha' should be set from 0 to 1")
        # else:
        #     self.alpha = alpha

        super(ImageSoftmaxEngine, self).__init__(datamanager,
                                                 self.val,
                                                 self.self_sup,
                                                 self.lambda_id,
                                                 self.lambda_ss,
                                                 use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

        if self.self_sup:
            # Multi-GPU attribute access
            if isinstance(self.model, nn.DataParallel):
                num_jig_classes = self.model.module.num_jig_classes
            else:
                num_jig_classes = self.model.num_jig_classes

            self.jig_criterion = CrossEntropyLoss(
                num_classes=num_jig_classes,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth
            )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if self.self_sup:
            outputs, jig_outputs, jig_labels = self.model(imgs)
            # print("self sup output length: ", len(outputs))
            # print("self sup output: ", outputs)
            jig_loss = self.compute_loss(self.criterion, jig_outputs, jig_labels)
        else:
            outputs, _ = self.model(imgs)

        # print(len(outputs))
        # print(outputs)
        if isinstance(outputs, (tuple, list)):
            loss = self.compute_loss(self.criterion, outputs[0], pids)
            for i in range(len(outputs) - 1):
                loss += self.compute_loss(self.criterion, outputs[i + 1], pids)
        else:
            loss = self.compute_loss(self.criterion, outputs, pids)
        # print(loss)

        self.optimizer.zero_grad()
        if self.self_sup:
            tot_loss = self.lambda_id * loss + self.lambda_ss * jig_loss
            tot_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        if self.self_sup:
            loss_summary = {
                'loss': loss.item(),
                'jig_loss': jig_loss.item(),
                'tot_loss': tot_loss.item(),
                # add accuracy for self sup task
                'acc': metrics.accuracy(outputs, pids)[0].item()

            }
        else:
            loss_summary = {
                'loss': loss.item(),
                'acc': metrics.accuracy(outputs, pids)[0].item()
            }

        return loss_summary
