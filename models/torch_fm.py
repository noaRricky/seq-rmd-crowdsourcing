from typing import List, Dict, Callable

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch_scatter import scatter_mean
from tqdm import tqdm

from utils import build_logger

logger = build_logger()


class TorchFM(nn.Module):
    def __init__(
            self,
            cat_dict: Dict[str, np.ndarray],
            pos_cat_names: List[str],
            neg_cat_names: List[str],
            num_dim: int = 10,
    ):
        super(TorchFM, self).__init__()

        self.cat_dict = cat_dict
        self.pos_cat_names = pos_cat_names
        self.neg_cat_names = neg_cat_names
        self.num_dim = num_dim
        self.emb_linear_layer = nn.ModuleList(
            [nn.Embedding(cat_dict[name].size, 1) for name in pos_cat_names])
        self.emb_factor_layer = nn.ModuleList([
            nn.Embedding(cat_dict[name].size, num_dim)
            for name in pos_cat_names
        ])

    def forward(self, pos_batch, neg_batch):
        # Linear terms
        pos_linear = self.compute_linear_term(pos_batch)
        neg_linear = self.compute_linear_term(neg_batch)

        # Interaction terms
        pos_factor = self.compute_factor_term(pos_batch)
        neg_factor = self.compute_factor_term(neg_batch)

        pos_preds = pos_linear + pos_factor
        neg_preds = neg_linear + neg_factor

        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds

    def compute_linear_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        linear_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_linear_layer)
        ]
        linear = T.sum(T.stack(linear_list), dim=0)

        return linear

    def compute_factor_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        emb_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_factor_layer)
        ]
        emb_mul = T.sum(T.stack(emb_list), dim=0)

        term_1 = T.pow(T.sum(emb_mul, dim=1, keepdim=True), 2)
        term_2 = T.sum(T.pow(emb_mul, 2), dim=1, keepdim=True)
        factor = 0.5 * (term_1 - term_2)
        return factor


class FMLearner(object):
    def __init__(self, model: TorchFM, optimzer: Callable,
                 train_dl: DataLoader, valid_dl: DataLoader,
                 test_dl: DataLoader):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.optimizer = optimzer

    def fit(
            self,
            epoch: int,
            lr: float,
            lr_decay_factor: float = 1,
            lr_decay_freq: int = 1000,
            linear_reg: float = 0.0,
            factor_reg: float = 0.0,
    ):
        op = self.optimizer(self.model.parameters, lr=lr)
        schedular = optim.lr_scheduler.StepLR(op,
                                              lr_decay_freq,
                                              gamma=lr_decay_factor)
        writer = SummaryWriter()
        global_step = 0

        for cur_epoch in tqdm(range(epoch)):
            print('Epoch: {}'.format(cur_epoch))
            for step, (pos_batch, neg_batch) in enumerate(self.train_dl):

                # zero the parameter gradients
                op.zero_grad()

                pos_preds, neg_preds = self.model(pos_batch, neg_batch)
                bprloss = self.criterion(pos_preds,
                                         neg_preds,
                                         linear_reg=linear_reg,
                                         factor_reg=factor_reg)

                bprloss.backward()
                op.step()
                schedular.step()  # type: ignore

                writer.add_scalar('loss',
                                  bprloss.item(),
                                  global_step=cur_epoch)

                with T.no_grad():
                    auc = self.metric(pos_batch, pos_preds, neg_preds)
                writer.add_scalar('train_accuracy',
                                  auc.item(),
                                  global_step=cur_epoch)
                print("epoch: {}, train loss: {}, train auccurcy: {}".format(
                    cur_epoch, bprloss.item(), auc.item()))
            for step, (pos_batch, neg_batch) in enumerate(self.test_dl):
                with T.no_grad():
                    pos_preds, neg_preds = self.model(pos_batch, neg_batch)
                    valid_bprloss = self.criterion(pos_batch,
                                                   neg_preds,
                                                   linear_reg=linear_reg,
                                                   factor_reg=factor_reg)
                    valid_auc = self.metric(pos_batch, pos_preds, neg_preds)
                writer.add_scalar('valid_loss',
                                  valid_bprloss.item(),
                                  global_step=cur_epoch)
                writer.add_scalar('valid_auc',
                                  valid_auc.item(),
                                  global_step=cur_epoch)
                print("epoch: {}, valid loss: {}, valid auccurcy: {}".format(
                    cur_epoch, valid_bprloss.item(), valid_auc.item()))

        writer.close()

    def compute_l2_term(self, linear_reg: float = 0.0,
                        factor_reg: float = 0.0) -> T.Tensor:
        l2_term = T.zeros(1)
        for emb in self.model.emb_linear_layer:
            l2_term += linear_reg * T.sum(T.pow(emb.weight.data, 2))
        for emb in self.model.emb_factor_layer:
            l2_term += factor_reg * T.sum(T.pow(emb.weight.data, 2))
        return l2_term

    def criterion(
            self,
            pos_preds: T.Tensor,
            neg_preds: T.Tensor,
            linear_reg: float = 0.0,
            factor_reg: float = 0.0,
    ) -> T.Tensor:
        l2_reg = self.compute_l2_term(linear_reg=linear_reg,
                                      factor_reg=factor_reg)
        bprloss = T.sum(
            T.log(1e-10 + T.sigmoid(pos_preds - neg_preds))) - l2_reg
        bprloss = -bprloss
        return bprloss

    def metric(self, pos_batch: T.Tensor, pos_preds: T.Tensor,
               neg_preds: T.Tensor) -> T.Tensor:

        binary_ranks = (pos_preds - neg_preds) > 0
        binary_ranks = binary_ranks.to(T.float)
        users_index = pos_batch[:, 0]
        users, user_counts = T.unique(users_index, return_counts=True)
        auc_per_user = scatter_mean(binary_ranks, users_index)
        auc = T.sum(auc_per_user) / users.size(0)
        return auc
