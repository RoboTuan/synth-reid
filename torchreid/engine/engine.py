from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import wandb
import sys

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results, mkdir_if_missing
)
from torchreid.losses import DeepSupervision
from torchreid.utils.torchtools import plot_grad_flow


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager):
            an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
            self,
            datamanager,
            val=False,
            lambda_id=1,
            lambda_ss=1,
            use_gpu=True,
            adversarial=False
    ):
        self.datamanager = datamanager
        self.val = val
        self.adversarial = adversarial
        self.lambda_id = lambda_id
        self.lambda_ss = lambda_ss
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        if self.val:
            self.val_loader = self.datamanager.val_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.epoch = 0

        self.model = None
        self.model_name = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, rank1, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            if self.adversarial:
                # TODO: vedere che altro salvare per adversarial
                save_checkpoint(
                    {
                        'state_dict': self._models[name].state_dict(),
                        'epoch': epoch + 1,
                        'optimizer': self._optims[name].state_dict(),
                        'scheduler': self._scheds[name].state_dict()
                    },
                    osp.join(save_dir, name),
                    is_best=is_best
                )
            else:
                save_checkpoint(
                    {
                        'state_dict': self._models[name].state_dict(),
                        'epoch': epoch + 1,
                        'rank1': rank1,
                        'optimizer': self._optims[name].state_dict(),
                        'scheduler': self._scheds[name].state_dict()
                    },
                    osp.join(save_dir, name),
                    is_best=is_best
                )

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        lrs = []
        for name in names:
            lrs.append((str(name + '_lr: '), self._optims[name].param_groups[-1]['lr']))
        return lrs

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        eval_flip=False
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional):
                number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0.
                ``fixbase_epoch`` is counted in ``max_epoch``.
            open_layers (str or list, optional):
                layers (attribute names) open for training.
            start_eval (int, optional):
                from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional):
                evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional):
                if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional):
                distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional):
                performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional):
                visualizes ranked results. Default is False.
                It is recommended to enable ``visrank`` when
                ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional):
                top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional):
                use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when
                using cuhk03 classic split.
            ranks (list, optional):
                cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional):
                uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            eval_flip (bool, optional):
                flip the image during evaluation time. Default if False.
                This is only enabled when test_only=True.
        """

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                flip=eval_flip,
                test_only=True
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
                save_dir=save_dir
            )

            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch + 1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                if self.adversarial:
                    # Don't save rank for adversarial
                    rank1 = self.test(
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        flip=eval_flip
                    )
                    self.save_model(epoch=self.epoch, save_dir=save_dir, rank1=rank1)
                    # self.save_model(epoch=self.epoch, save_dir=save_dir, rank1=None)
                else:
                    rank1 = self.test(
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        flip=eval_flip
                    )
                    self.save_model(self.epoch, rank1, save_dir)

        if self.max_epoch > 0:
            # When adversarial is True, save model without testing
            if self.adversarial:
                print('=> Final test')
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    flip=eval_flip
                )
                # self.save_model(epoch=self.epoch, save_dir=save_dir, rank1=None)
                self.save_model(epoch=self.epoch, save_dir=save_dir, rank1=rank1)
            else:
                print('=> Final test')
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    flip=eval_flip
                )
                self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None, save_dir=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        if self.adversarial:
            train_t_iterator = iter(self.datamanager.train_loader_t)

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            self.set_model_mode('train')
            if self.adversarial:
                try:
                    other_data = next(train_t_iterator)
                except StopIteration:
                    train_t_iterator = iter(self.datamanager.train_loader_t)
                    other_data = next(train_t_iterator)

            data_time.update(time.time() - end)

            if self.adversarial:
                loss_summary, imgs_S, imgs_R, r2r = self.forward_backward_adversarial(self.batch_idx, data, other_data)
            else:
                loss_summary = self.forward_backward(data)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            n_iter = self.epoch * self.num_batches + self.batch_idx

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                    ),
                    # modifies to .8f format for smaller learning rates
                    '\t '.join([str(lr[0] + str("%.8f" % lr[1])) for lr in self.get_current_lr()])
                )
                if self.adversarial:
                    # _ = self.test(
                    #     dist_metric='euclidean',
                    #     normalize_feature=False,
                    #     visrank=False,
                    #     save_dir=save_dir,
                    #     use_metric_cuhk03=False,
                    #     ranks=[1, 5, 10, 20],
                    #     flip=False
                    # )
                    if imgs_S.shape[0] != 1:
                        img_S = torch.unsqueeze(imgs_S[0], 0)
                        img_R = torch.unsqueeze(imgs_R[0], 0)
                        r2r = torch.unsqueeze(r2r[0], 0)
                    else:
                        img_S = imgs_S
                        img_R = imgs_R
                        r2r = r2r

                    for model_name, model in self.models.items():
                        plot_grad_flow(model.module.named_parameters(), model_name, n_iter)
                    # plot_grad_flow(self.models['mlp'].module.named_parameters(), 'mlp', n_iter)
                    # sys.exit()
                    with torch.no_grad():
                        # n_iter = self.epoch * self.num_batches + self.batch_idx
                        # print(n_iter)
                        self.set_model_mode('eval')
                        a_real_test = img_S
                        b_real_test = img_R
                        # print(self.models)
                        # print(self._models)
                        b_fake_test = self.models['generator'](a_real_test)

                        pic = (torch.cat([
                            a_real_test, b_fake_test, b_real_test, r2r],
                            dim=0).data + 1) / 2.0
                        # pic = torchvision.utils.make_grid(pic, nrow=2)

                        # only len of train loader and not max between
                        # this and the other dataloader, since we're iterating on this
                        caption = "Epoch_({})_({}of{}).jpg".format(
                            self.epoch + 1, self.batch_idx + 1, len(self.train_loader)
                        )
                        # tag = "Epoch_({}).jpg".format(
                        #     self.epoch + 1
                        # )
                        wandb.log({"media/images": wandb.Image(pic, caption=caption)}, step=n_iter + 1)
                        # self.writer.add_image(tag, torchvision.utils.make_grid(pic), global_step=self.epoch)
                        # save_img_dir = save_dir + '/sample_images_while_training'
                        # mkdir_if_missing(save_img_dir)
                        # torchvision.utils.save_image(pic, '%s/%s' % (save_img_dir, caption), nrow=2)

            if self.writer is not None:
                # n_iter = self.epoch * self.num_batches + self.batch_idx
                # print(n_iter)
                wandb.log({'Train_info/Epochs': self.epoch + 1}, step=n_iter + 1)
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                wandb.log({'Train_info/time': batch_time.avg}, step=n_iter + 1)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                wandb.log({'Train_info/data': data_time.avg}, step=n_iter + 1)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                    wandb.log({'Train_loss/' + name: meter.avg}, step=n_iter + 1)
                for lr in self.get_current_lr():
                    self.writer.add_scalar(('Train/lr_' + lr[0]), lr[1], n_iter)
                    wandb.log({'Train_lr/lr_' + lr[0]: lr[1]}, step=n_iter + 1)

            end = time.time()
        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def forward_backward_adversarial(self, data):
        raise NotImplementedError

    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        flip=False,
        test_only=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both
            image- and video-reid. In general, a subclass of Engine only
            needs to re-implement ``extract_features()`` and
            ``parse_data_for_eval()`` (most of the time), but not a must.
            Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        if self.val and not test_only:
            # There is 1 validation dataloader
            targets = list(self.val_loader.keys())
            self.phase = 'Val'
        elif test_only and self.val:
            targets = list(self.test_loader.keys())
            self.phase = 'Test'
        else:
            targets = list(self.test_loader.keys())
            self.phase = 'Test'
        print(self.phase)

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            if self.phase == 'Val':
                print("Evaluating on validation data")
                query_loader = self.val_loader[name]['query']
                # print(len(query_loader))
                gallery_loader = self.val_loader[name]['gallery']
                # print(len(gallery_loader))
            else:
                print("Evaluating on testing data")
                query_loader = self.test_loader[name]['query']
                gallery_loader = self.test_loader[name]['gallery']

            cmc, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                flip=flip
            )

            rank1 = cmc[0]
            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                for r in ranks:
                    wandb.log({f'{self.phase}/{name}/Rank{r}': cmc[r - 1]}, step=n_iter + 1)
                self.writer.add_scalar(f'{self.phase}/{name}/rank1', rank1, self.epoch)
                self.writer.add_scalar(f'{self.phase}/{name}/mAP', mAP, self.epoch)
                wandb.log({f'{self.phase}/{name}/mAP': mAP}, step=n_iter + 1)

        return rank1

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        flip=False
    ):
        batch_time = AverageMeter()

        if flip is True:
            print("Adding flipped imgs...")

        def fliplr(img):
            '''flip horizontal'''
            inv_idx = torch.arange(
                img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
            img_flip = img.index_select(3, inv_idx)
            return img_flip

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)

                if flip:
                    imgs = fliplr(imgs)
                    features_flip = self.extract_features(imgs)
                    # print(features)
                    # print(features_flip)
                    features += features_flip
                    # print(features)
                    # print(features.shape)
                # print(features)
                # print(features.shape)
                batch_time.update(time.time() - end)
                features = features.cpu().clone()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        # return cmc[0], mAP
        return cmc, mAP

    def compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            # print("self sup targets: ", targets)
            # print("self sup outputs: ", outputs)
            # print(type(targets), type(targets[0]))
            # print(type(outputs), type(outputs[0]))
            # print(targets[0])
            # sys.exit()
            loss = criterion(outputs, targets)
        return loss

    def extract_features(self, input):
        if self.adversarial:
            features = self.models['generator'](input, feat_extractor=True)
            global_features = self.models['id_net'](features)
            return global_features

        else:
            return self.model(input)

    # def get_output_shape(self, image_dim):
    #     return self.model(torch.rand(*(image_dim))).data.shape

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        return imgs, pids

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            # print(self.model_name)
            open_specified_layers(self.model_name, model, open_layers)
        else:
            open_all_layers(model)
