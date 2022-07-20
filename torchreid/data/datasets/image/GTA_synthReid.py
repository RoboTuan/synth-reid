import glob
import re
from collections import defaultdict
import random

import os.path as osp
from typing import Tuple, Union
import numpy as np

from ..dataset import ImageDataset


class GTA_synthReid(ImageDataset):
    """
    GTA_synthReid
    """

    dataset_dir = 'GTA_synthReid'

    # The option relabel is used only for testing purposes
    def __init__(self,
                 root='/mnt/data2/defonte_data/PersonReid_datasets/',
                 val=False,
                 relabel=True,
                 n_samples=50,
                 **kwargs) -> None:
        # super(GTA_synthReid, self).__init__()

        # set_random_seed()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_val_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.val = val
        self.relabel = relabel
        self.n_samples = n_samples
        # print(self.val)

        # PUT RELABEL TO FALSE IF VALIDATION SPLIT IS PERFORMED,
        # OTHERWISE THE LABELS WON'T BE CORRECT.
        # RELABEL AFTER THE TRAIN/VAL SPLIT.
        if self.val is True:
            # keep relabel False here
            train_val, train_val_pids = self._process_dir(self.train_val_dir, subsample=True, relabel=False)
        else:
            train_val, train_val_pids = self._process_dir(self.train_val_dir, subsample=True, relabel=self.relabel)

        query, _ = self._process_dir(self.query_dir, relabel=False)
        gallery, _ = self._process_dir(self.gallery_dir, subsample=True, relabel=False)

        self.train_val_pids = train_val_pids
        self.train_val = train_val
        self.query = query
        self.gallery = gallery

        if self.val is True:
            train, val_gallery = self._train_val_split()
            self.train = train
            self.val_gallery = val_gallery

            val_query = self._prepare_val()
            self.val_query = val_query

            super(GTA_synthReid, self).__init__(train, query, gallery, self.val, val_query, val_gallery, **kwargs)

        else:
            super(GTA_synthReid, self).__init__(train_val, query, gallery, **kwargs)

    def _check_before_run(self) -> None:
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_val_dir):
            raise RuntimeError("'{}' is not available".format(self.train_val_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, subsample=False, relabel=False) -> Tuple[list, np.array]:
        img_paths = glob.glob(osp.join(dir_path, '*.jpeg'))
        pattern = re.compile(r'([\d]+)_s([dn])([\d]+)c(\d{2})')

        if relabel is True:
            pid_container = set()
            for img_path in img_paths:
                pid, _, _, _ = map(lambda x: convertToType(x), pattern.search(img_path).groups())
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        pids = set()
        dataset = []
        for img_path in img_paths:
            pid, _, _, camid = map(lambda x: convertToType(x), pattern.search(img_path).groups())
            assert 1 <= camid <= 19  # original camera ids are from 1 to 19
            camid -= 1  # index starts from 0
            if relabel is True:
                pid = pid2label[pid]
            pids.add(pid)  # add ids (relabelled or not) to set
            dataset.append((img_path, pid, camid))

        if subsample is True:
            pid_images = defaultdict(list)
            for pid in pids:
                for image in dataset:
                    if image[1] == pid:
                        pid_images[pid].append(image)
            dataset = []
            for values in pid_images.values():
                if len(values) <= self.n_samples:
                    dataset.extend(values)
                else:
                    samples = random.sample(values, self.n_samples)
                    dataset.extend(samples)

        return dataset, np.array(list(pids))

    def _train_val_split(self):
        train_dataset = []
        val_dataset = []
        n_val_pids = 50  # hard coded
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
        pattern = re.compile(r'([\d]+)_s([dn])([\d]+)c(\d{2})')

        for image in self.val_gallery:
            img_path = image[0]
            pid = image[1]
            cam_id = pattern.search(img_path).group(4)
            if pid not in pid_cams:
                pid_cams[pid].append(cam_id)
                val_query.append(image)
            elif cam_id not in pid_cams[pid]:
                pid_cams[pid].append(cam_id)
                val_query.append(image)
        # print(pid_cams)
        return val_query


def convertToType(string) -> Union[int, str]:
    if string.isdigit():
        return int(string)
    else:
        return string
