# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
import time
from itertools import cycle

import numpy as np
import torch
import torch.utils.data as dutils
from loguru import logger
from collections import defaultdict
import random


class EqualSampler(dutils.Sampler):
    def __init__(self, datasets, batch_size=1, ratio_2d=0.5, shuffle=False):
        super(EqualSampler, self).__init__(datasets)
        self.num_datasets = len(datasets)
        self.ratio_2d = ratio_2d

        self.shuffle = shuffle
        self.dset_sizes = {}
        self.elements_per_index = {}
        self.only_2d = {}
        self.offsets = {}
        start = 0
        for dset in datasets:
            self.dset_sizes[dset.name()] = len(dset)
            self.offsets[dset.name()] = start
            self.only_2d[dset.name()] = dset.only_2d()
            self.elements_per_index[
                dset.name()] = dset.get_elements_per_index()

            start += len(dset)

        if ratio_2d < 1.0 and sum(self.only_2d.values()) == len(self.only_2d):
            raise ValueError(
                f'Invalid 2D ratio value: {ratio_2d} with only 2D data')

        self.length = sum(map(lambda x: len(x), datasets))

        self.batch_size = batch_size
        self._can_reuse_batches = False
        logger.info(self)

    def __repr__(self):
        msg = 'EqualSampler(batch_size={}, shuffle={}, ratio_2d={}\n'.format(
            self.batch_size, self.shuffle, self.ratio_2d)
        for dset_name in self.dset_sizes:
            msg += '\t{}: {}, only 2D is {}\n'.format(
                dset_name, self.dset_sizes[dset_name],
                self.only_2d[dset_name])

        return msg + ')'

    def _prepare_batches(self):
        batch_idxs = []

        dset_idxs = {}
        for dset_name, dset_size in self.dset_sizes.items():
            if self.shuffle:
                dset_idxs[dset_name] = cycle(
                    iter(torch.randperm(dset_size).tolist()))
            else:
                dset_idxs[dset_name] = cycle(range(dset_size))

        num_batches = int(round(self.length / self.batch_size))
        for bidx in range(num_batches):
            curr_idxs = []
            num_samples = 0
            num_2d_only = 0
            max_num_2d = int(self.batch_size * self.ratio_2d)
            idxs_add = defaultdict(lambda: 0)
            while num_samples < self.batch_size:
                for dset_name in dset_idxs:
                    # If we already have self.ratio_2d * batch_size items with
                    # 2D annotations then ignore this dataset for now
                    if num_2d_only >= max_num_2d and self.only_2d[dset_name]:
                        continue
                    try:
                        curr_idxs.append(
                            next(dset_idxs[dset_name]) +
                            self.offsets[dset_name])
                        num_samples += self.elements_per_index[dset_name]
                        # If the dataset has only 2D annotations increase the
                        # count
                        num_2d_only += (self.elements_per_index[dset_name] *
                                        self.only_2d[dset_name])
                        idxs_add[dset_name] += (
                            self.elements_per_index[dset_name])
                    finally:
                        pass
                    if num_samples >= self.batch_size:
                        break

            curr_idxs = np.array(curr_idxs)
            if self.shuffle:
                np.random.shuffle(curr_idxs)
            batch_idxs.append(curr_idxs)
        return batch_idxs

    def __len__(self):
        if not hasattr(self, '_batch_idxs'):
            self._batch_idxs = self._prepare_batches()
            self._can_reuse_bathces = True
        return len(self._batch_idxs)

    def __iter__(self):
        if self._can_reuse_batches:
            batch_idxs = self._batch_idxs
            self._can_reuse_batches = False
        else:
            batch_idxs = self._prepare_batches()

        self._batch_idxs = batch_idxs
        return iter(batch_idxs)


class SequenceSampler(dutils.Sampler):
    def __init__(self, datasets, batch_size=1, shuffle=True):
        super(SequenceSampler, self).__init__(datasets)
        self.num_datasets = len(datasets)
        self.dset_sizes = {}
        self.elements_per_index = {}
        self.offsets = {}

        start = 0
        self.datasets = {}
        for dset in datasets:
            self.datasets[dset.name()] = dset
            self.dset_sizes[dset.name()] = len(dset)
            self.offsets[dset.name()] = start
            self.elements_per_index[
                dset.name()] = dset.get_elements_per_index()

            start += len(dset)

        self.shuffle = shuffle
        self.batch_size = batch_size
        self._can_reuse_batches = False
        self.length = sum(map(lambda x: len(x.get_subseq_start_end_indices()), datasets))
        logger.info(self)

    def __repr__(self):
        msg = 'SequenceSampler(batch_size={}, \n'.format(
            self.batch_size)
        for dset_name in self.dset_sizes:
            msg += '\t{}: {}\n'.format(
                dset_name, self.dset_sizes[dset_name])

        return msg + ')'

    def _prepare_batches(self):
        batch_idxs = []

        sequences_per_dataset = defaultdict(cycle)
        for dset_name in self.datasets.keys():
            subseq_list = self.datasets[dset_name].get_subseq_start_end_indices()
            if self.shuffle:
                random.shuffle(subseq_list)
            sequences_per_dataset[dset_name] = cycle(iter(subseq_list))

        num_batches = int(round(self.length / self.batch_size))
        for bidx in range(num_batches):
            curr_idxs = []
            num_sequences = 0
            while num_sequences < self.batch_size:
                for dset_name in sequences_per_dataset.keys():
                    curr_seq = next(sequences_per_dataset[dset_name])

                    curr_idxs += [idx + self.offsets[dset_name] for idx in curr_seq]
                    num_sequences += 1

                    if num_sequences == self.batch_size:
                        break

            if num_sequences == self.batch_size:
                batch_idxs.append(np.array(curr_idxs))

        return batch_idxs

    def __len__(self):
        if not hasattr(self, '_batch_idxs'):
            self._batch_idxs = self._prepare_batches()
            self._can_reuse_bathces = True
        return len(self._batch_idxs)

    def __iter__(self):
        if self._can_reuse_batches:
            batch_idxs = self._batch_idxs
            self._can_reuse_batches = False
        else:
            batch_idxs = self._prepare_batches()

        self._batch_idxs = batch_idxs
        return iter(batch_idxs)
