from __future__ import division, print_function, absolute_import

import torch.nn as nn

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, CenterLoss

from ..engine import Engine


class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
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
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model_name,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        weight_c=0,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        val=False,
        self_sup=False
    ):
        self.val = val
        self.self_sup = self_sup
        if self.self_sup:
            raise ValueError("Self sup not yet implemented for triplet loss")

        super(ImageTripletEngine, self).__init__(datamanager=datamanager,
                                                 val=self.val,
                                                 self_sup=self.self_sup,
                                                 use_gpu=use_gpu)

        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0 and weight_c >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_c = weight_c

        if not self.self_sup:
            if isinstance(self.model, nn.DataParallel):
                try:
                    feature_dim = self.model.module.feature_dim
                except AttributeError as e:
                    print(e)
                    print("Trying with attribute 'in_planes'...")
                    feature_dim = self.model.module.in_planes
                    print("... working!")
            else:
                try:
                    feature_dim = self.model.feature_dim
                except AttributeError as e:
                    print(e)
                    print("Trying with attribute 'in_planes'...")
                    feature_dim = self.model.in_planes
                    print("... working!")

        # print("loss wieghts: ", self.weight_t, self.weight_x, self.weight_c)
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_c = CenterLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            feat_dim=feature_dim
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
                # TODO: check if label smooth should be applied also to sel sup task
                # label_smooth=label_smooth
            )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        # 'features' are the feature maps after the global avg pool in resnet
        if self.self_sup:
            outputs, features, jig_outputs, jig_labels = self.model(imgs)
        else:
            outputs, features, _ = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.self_sup:
            pass
        else:
            if self.weight_t > 0:
                loss_t = self.compute_loss(self.criterion_t, features, pids)
                loss += self.weight_t * loss_t
                loss_summary['loss_t'] = loss_t.item()

            if self.weight_c > 0:
                loss_c = self.compute_loss(self.criterion_c, features, pids)
                loss += self.weight_c * loss_c
                loss_summary['loss_c'] = loss_c.item()

            # If the weight of the softmax is 0,
            # the output of classifier is not included in
            # the computational graph and accuracy is not computed
            if self.weight_x > 0:
                loss_x = self.compute_loss(self.criterion_x, outputs, pids)
                loss += self.weight_x * loss_x
                loss_summary['loss_x'] = loss_x.item()
                loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

            assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
