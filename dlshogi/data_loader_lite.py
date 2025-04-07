import numpy as np
import torch

from dlshogi.common import *
from dlshogi import cppshogi

import os
from concurrent.futures import ThreadPoolExecutor

import logging


FEATURES1_LITE_NUM = 1 + 2 * (PIECETYPE_NUM + MAX_ATTACK_NUM)


class DataLoader:
    @staticmethod
    def load_files(files, logger=logging):
        data = []
        for path in files:
            if os.path.exists(path):
                logger.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logger.warn("{} not found, skipping".format(path))
        return np.concatenate(data)

    def __init__(self, data, batch_size, device, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features1 = torch.empty(
            (batch_size, 9 * 9, FEATURES1_LITE_NUM), dtype=torch.long, pin_memory=True
        )
        self.torch_features2 = torch.empty(
            (batch_size, FEATURES2_NUM), dtype=torch.long, pin_memory=True
        )
        self.torch_result = torch.empty(
            (batch_size, 1), dtype=torch.float32, pin_memory=True
        )
        self.torch_value = torch.empty(
            (batch_size, 1), dtype=torch.float32, pin_memory=True
        )

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def mini_batch(self, hcpevec):
        cppshogi.hcpe_decode_lite(
            hcpevec, self.features1, self.features2, self.result, self.value
        )

        if self.device.type == "cpu":
            return (
                self.torch_features1.clone(),
                self.torch_features2.clone(),
                self.torch_result.clone(),
                self.torch_value.clone(),
            )
        else:
            return (
                self.torch_features1.to(self.device),
                self.torch_features2.to(self.device),
                self.torch_result.to(self.device),
                self.torch_value.to(self.device),
            )

    def sample(self):
        return self.mini_batch(
            np.random.choice(self.data, self.batch_size, replace=False)
        )

    def pre_fetch(self):
        hcpevec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result


# 評価値から勝率への変換
def score_to_value(score, a):
    return 1.0 / (1.0 + np.exp(-score / a))


class Hcpe3DataLoader(DataLoader):
    @staticmethod
    def load_files(
        files,
        use_average=False,
        use_evalfix=False,
        temperature=1.0,
        patch=None,
        cache=None,
        logger=logging,
    ):
        # キャッシュが存在する場合、キャッシュから読み込む
        if cache and os.path.isfile(cache):
            logger.info("Load cache {}".format(cache))
            cache_len = cppshogi.hcpe3_load_cache(cache)
            return cache_len, cache_len

        if use_evalfix:
            from scipy.optimize import curve_fit

        actual_len = 0
        for path in files:
            if os.path.exists(path):
                if use_evalfix:
                    eval, result = cppshogi.hcpe3_prepare_evalfix(path)
                    if (eval == 0).all():
                        a = 0
                        logger.info("{}, skip evalfix".format(path))
                    else:
                        popt, _ = curve_fit(score_to_value, eval, result, p0=[300.0])
                        a = popt[0]
                        logger.info("{}, a={}".format(path, a))
                else:
                    a = 0
                    logger.info(path)
                sum_len, len_ = cppshogi.load_hcpe3(path, use_average, a, temperature)
                if len_ == 0:
                    raise RuntimeError("read error {}".format(path))
                actual_len += len_
            else:
                logger.warn("{} not found, skipping".format(path))
        if patch:
            # パッチを当てる
            patch_sum_len, patch_add_len = cppshogi.hcpe3_patch_with_hcpe(patch)
            logger.info(
                "Patch with {}, patched num = {}, added num = {}".format(
                    patch, patch_sum_len - patch_add_len, patch_add_len
                )
            )
        if cache:
            # キャッシュ作成
            logger.info("Create cache {}".format(cache))
            cppshogi.hcpe3_create_cache(cache)
            cache_len = cppshogi.hcpe3_load_cache(cache)
            assert sum_len == cache_len
        return sum_len, actual_len

    def __init__(self, data, batch_size, device, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.torch_features1 = torch.empty(
            (batch_size, 9 * 9, FEATURES1_LITE_NUM), dtype=torch.long, pin_memory=True
        )
        self.torch_features2 = torch.empty(
            (batch_size, FEATURES2_NUM), dtype=torch.long, pin_memory=True
        )
        self.torch_result = torch.empty(
            (batch_size, 1), dtype=torch.float32, pin_memory=True
        )
        self.torch_value = torch.empty(
            (batch_size, 1), dtype=torch.float32, pin_memory=True
        )

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def mini_batch(self, index):
        cppshogi.hcpe3_decode_lite(
            index,
            self.features1,
            self.features2,
            self.result,
            self.value,
        )

        return (
            self.torch_features1.to(self.device),
            self.torch_features2.to(self.device),
            self.torch_result.to(self.device),
            self.torch_value.to(self.device),
        )
