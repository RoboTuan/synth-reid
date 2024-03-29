from __future__ import division, print_function, absolute_import
from collections import defaultdict
import torch
from torchreid.data.datasets.dataset import Dataset

from torchreid.data.sampler import build_train_sampler
from torchreid.data.datasets import init_image_dataset, init_video_dataset
from torchreid.data.transforms import build_transforms


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        seed=10,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False,
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width
        self.seed = seed

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        load_train_targets (bool, optional): construct train-loader for target datasets.
            Default is False. This is useful for domain adaptation research.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        train_sampler_t (str, optional): sampler for target train loader. Default is RandomSampler.
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader

        # return train loader of target data
        train_loader_t = datamanager.train_loader_t
    """
    data_type = 'image'

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler='RandomSampler',
        train_sampler_t='RandomSampler',
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False,
        verbose=True,
        # add option for validation set
        val=False,
        adversarial=False,
        relabel=True,
        seed=10,
        n_samples=50  # taking only #n_samples images for GTA_synthReid
    ):

        super(ImageDataManager, self).__init__(
            seed=seed,
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        self.verbose = verbose
        self.adversarial = adversarial
        if self.adversarial is True:
            self.val = False
            self.adv_val = val
            self.val_sources = self.targets
        else:
            self.val = val
            self.adv_val = False
            self.val_sources = self.sources
        self.relabel = relabel
        self.n_samples = n_samples
        self.batch_size_train = batch_size_train
        # print(self.val)

        self.trainsets = defaultdict(Dataset)
        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                self.seed,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode='train',
                combineall=combineall,
                root=root,
                verbose=self.verbose,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                val=self.val,
                relabel=self.relabel,
                n_samples=self.n_samples
            )
            self.trainsets[name] = trainset_
            trainset.append(trainset_)

        trainset = sum(trainset)
        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams
        # print(self.trainsets)
        # print(trainset.__getitem__(0))
        # print(trainset.__getitem__(1))
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            # worker_init_fn=np.random.seed(0),
            pin_memory=self.use_gpu,
            drop_last=True
        )

        self.train_loader_t = None
        if load_train_targets:
            # check if sources and targets are identical
            assert len(set(self.sources) & set(self.targets)) == 0, \
                'sources={} and targets={} must not have overlap'.format(self.sources, self.targets)

            print('=> Loading train (target) dataset')
            print('Sampler for this dataset is RandomSampler')
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    self.seed,
                    transform=self.transform_tr,
                    k_tfm=k_tfm,
                    mode='train',
                    combineall=False,  # only use the training data
                    root=root,
                    verbose=self.verbose,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    val=self.adv_val,
                    relabel=self.relabel,
                    n_samples=self.n_samples
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    "RandomSampler",
                    batch_size=batch_size_train,
                    num_instances=num_instances,
                    num_cams=num_cams,
                    num_datasets=num_datasets
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )

        if self.val is True or self.adv_val is True:
            print('=> Loading validataion dataset')
            self.val_loader = {
                name: {
                    'query': None,
                    'gallery': None
                }
                for name in self.val_sources
            }
            self.val_dataset = {
                name: {
                    'query': None,
                    'gallery': None
                }
                for name in self.val_sources
            }
            for name in self.val_sources:
                # build query loader
                val_queryset = init_image_dataset(
                    name,
                    self.seed,
                    transform=self.transform_te,
                    mode='val_query',
                    combineall=False,
                    root=root,
                    verbose=self.verbose,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    val=True,
                    relabel=self.relabel,
                    n_samples=self.n_samples
                )
                # print(val_queryset)
                self.val_loader[name]['query'] = torch.utils.data.DataLoader(
                    val_queryset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=self.use_gpu,
                    drop_last=False
                )

                # build val_gallery loader
                val_galleryset = init_image_dataset(
                    name,
                    self.seed,
                    transform=self.transform_te,
                    mode='val_gallery',
                    combineall=False,
                    verbose=self.verbose,
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    val=True,
                    relabel=self.relabel,
                    n_samples=self.n_samples
                )

                self.val_loader[name]['gallery'] = torch.utils.data.DataLoader(
                    val_galleryset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=self.use_gpu,
                    drop_last=False
                )

                self.val_dataset[name]['query'] = val_queryset.val_query
                self.val_dataset[name]['gallery'] = val_galleryset.val_gallery
                # print(self.val_dataset[name]['query'].__getitem__(0))
                # print(self.val_dataset[name]['query'].__getitem__(1))
                # print(self.val_dataset[name]['gallery'].__getitem__(0))
                # print(self.val_dataset[name]['gallery'].__getitem__(1))

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                self.seed,
                transform=self.transform_te,
                mode='query',
                combineall=False,
                verbose=self.verbose,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                val=self.val,
                relabel=self.relabel,
                n_samples=self.n_samples
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                self.seed,
                transform=self.transform_te,
                mode='gallery',
                combineall=False,
                verbose=self.verbose,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                val=self.val,
                relabel=self.relabel,
                n_samples=self.n_samples
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source images   : {}'.format(len(trainset)))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        if load_train_targets:
            print(
                '  # target images   : {} (unlabeled)'.format(len(trainset_t))
            )
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')


class VideoDataManager(DataManager):
    r"""Video data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of tracklets in a training batch. Default is 3.
        batch_size_test (int, optional): number of tracklets in a test batch. Default is 3.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        seq_len (int, optional): how many images to sample in a tracklet. Default is 15.
        sample_method (str, optional): how to sample images in a tracklet. Default is "evenly".
            Choices are ["evenly", "random", "all"]. "evenly" and "random" will sample ``seq_len``
            images in a tracklet while "all" samples all images in a tracklet, where the batch size
            needs to be set to 1.

    Examples::

        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            batch_size_train=3,
            batch_size_test=3,
            seq_len=15,
            sample_method='evenly'
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader

    .. note::
        The current implementation only supports image-like training. Therefore, each image in a
        sampled tracklet will undergo independent transformation functions. To achieve tracklet-aware
        training, you need to modify the transformation functions for video reid such that each function
        applies the same operation to all images in a tracklet to keep consistency.
    """
    data_type = 'video'

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        batch_size_train=3,
        batch_size_test=3,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler='RandomSampler',
        seq_len=15,
        sample_method='evenly',
        seed=10,
        verbose=True,
        # TODO: add validation and test relabeling for video datasets
    ):

        super(VideoDataManager, self).__init__(
            seed=seed,
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        self.verbose = verbose

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_video_dataset(
                name,
                self.seed,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train,
            train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances,
            num_cams=num_cams,
            num_datasets=num_datasets
        )

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_video_dataset(
                name,
                self.seed,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_video_dataset(
                name,
                self.seed,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=self.verbose,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source             : {}'.format(self.sources))
        print('  # source datasets  : {}'.format(len(self.sources)))
        print('  # source ids       : {}'.format(self.num_train_pids))
        print('  # source tracklets : {}'.format(len(trainset)))
        print('  # source cameras   : {}'.format(self.num_train_cams))
        print('  target             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
