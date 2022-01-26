from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import numpy as np
from collections import defaultdict
from ..dataset import ImageDataset


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, val=False, relabel=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.val = val
        self.relabel = relabel
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_val_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_val_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        if self.val:
            train_val, train_val_pids = self.process_dir(self.train_val_dir, relabel=False)
        else:
            train_val, train_val_pids = self.process_dir(self.train_val_dir, relabel=self.relabel)
        query, _ = self.process_dir(self.query_dir, relabel=False)
        gallery, _ = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            # '[0]' since no pids are needed, only the list of images
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)[0]

        self.train_val_pids = np.array(list(train_val_pids))
        self.train_val = train_val
        self.query = query
        self.gallery = gallery
        if self.val:
            train, val_gallery = self._train_val_split()
            self.train = train
            self.val_gallery = val_gallery
            val_query = self._prepare_val()
            self.val_query = val_query
            super(Market1501, self).__init__(train, query, gallery, self.val, val_query, val_gallery, **kwargs)
        else:
            super(Market1501, self).__init__(train_val, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data, pid_container

    def _train_val_split(self):
        train_dataset = []
        val_dataset = []
        n_val_pids = 200  # hard coded
        # taking n_val_pids pids for validation
        val_pids = set(self.train_val_pids[np.random.choice(self.train_val_pids.shape[0],
                       n_val_pids, replace=False)])
        train_pids = dict()
        count = 0
        assert len(val_pids) == n_val_pids
        for image in self.train_val:
            if image[1] in val_pids:
                val_dataset.append(image)
            else:
                if image[1] not in train_pids:
                    train_pids[image[1]] = count
                    count += 1
                if self.relabel:
                    new_image = (image[0], train_pids[image[1]], image[2])
                else:
                    new_image = image
                train_dataset.append(new_image)

        return train_dataset, val_dataset

    def _prepare_val(self):
        # va_gallery is the same as val
        val_query = []
        pid_cams = defaultdict(list)
        pattern = re.compile(r'([-\d]+)_c(\d)')

        for image in self.val_gallery:
            img_path = image[0]
            pid = image[1]
            _, cam_id = pattern.search(img_path).groups()
            if pid not in pid_cams:
                pid_cams[pid].append(cam_id)
                val_query.append(image)
            elif cam_id not in pid_cams[pid]:
                pid_cams[pid].append(cam_id)
                val_query.append(image)
        # print(pid_cams)
        return val_query
