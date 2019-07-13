from typing import List, Dict, Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import build_logger

logger = build_logger()


class TorchFM(nn.Module):
    def __init__(self,
                 cat_dict: Dict[str, np.ndarray],
                 pos_cat_names: List[str],
                 neg_cat_names: List[str],
                 num_dim: int = 10,
                 linear_reg: float = 0.0,
                 factor_reg: float = 0.0):
        self.cat_dict = cat_dict
        self.pos_cat_names = pos_cat_names
        self.neg_cat_names = neg_cat_names
        self.num_dim = num_dim
        self.linear_reg = linear_reg
        self.factor_reg = factor_reg
        self.emb_linear_layer = nn.ModuleList(
            [nn.Embedding(cat_dict[name].size, 1) for name in pos_cat_names])
        self.emb_factor_layer = nn.ModuleList(
            [nn.Embedding(cat_dict[name], num_dim) for name in pos_cat_names])

    def forward(self, pos_batch, neg_batch):
        batch_size, _ = pos_batch.shape

        # Linear terms
        pos_linear = self.compute_linear_term(pos_batch)
        neg_linear = self.compute_linear_term(neg_batch)

        # Interaction terms
        pos_factor = self.compute_factor_term(pos_batch)
        neg_factor = self.compute_factor_term(neg_batch)

        pos_preds = pos_linear + pos_factor
        neg_preds = neg_linear + neg_factor

        l2_reg = self.compute_l2_term()

    def compute_linear_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        linear_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_linear_layer)
        ]
        linear = T.zeros(batch_size, 1)
        for item in linear_list:
            linear += item

        return linear

    def compute_factor_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        emb_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_factor_layer)
        ]
        emb_mul = T.zeros(batch_size, self.num_dim)
        for item in emb_list:
            emb_mul += item

        term_1 = T.pow(T.sum(emb_mul, dim=1, keepdim=True), 2)
        term_2 = T.sum(T.pow(emb_mul, 2), dim=1, keepdim=True)
        factor = 0.5 * (term_1 - term_2)
        return factor

    def compute_l2_term(self) -> T.Tensor:
        l2_term = T.zeros(1)
        for emb in self.emb_linear_layer:
            l2_term += self.linear_reg * T.sum(T.pow(emb.weight.data, 2))
        for emb in self.emb_factor_layer:
            l2_term += self.factor_reg * T.sum(T.pow(emb.weight.data, 2))
        return l2_term


class FMLearner(object):
    def __init__(self, model: nn.Module, optimzer: Callable,
                 train_dl: DataLoader, valid_dl: DataLoader,
                 test_dl: DataLoader):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.optimizer = optimzer

    def fit(self, epoch: int, lr: float):
        op = self.optimizer(self.model.parameters, lr=lr)

        for i in tqdm(range(epoch)):
            for step, (pos_batch, neg_batch) in enumerate(self.train_dl):
                pass
