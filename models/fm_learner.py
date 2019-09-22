from typing import List, Dict, Callable, Optional

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import build_logger
from datasets.torch_movielen import TorchMovielen10k
from .torch_fm import TorchFM, TorchTransFM

logger = build_logger()


class FMLearner(object):
    def __init__(
            self,
            model: TorchFM,
            op: Optimizer,
            schedular: _LRScheduler,
            databunch: TorchMovielen10k,
    ) -> None:

        ds_types = ['train', 'valid', 'test']
        # Genrate accuracy per user to compute
        user_size = databunch.user_size
        device = databunch.device

        self.dl_dict = {ds: databunch.get_dataloader(ds) for ds in ds_types}
        self.model = model.to(device)
        self._op = op
        self._schedular = schedular
        self.hit_per_user: T.Tensor = T.zeros(user_size)
        self.user_counts: T.Tensor = T.zeros(user_size)
        self.best_val_auc = 0.
        self.best_epoch = 0.

    def fit(
            self,
            epoch: int,
            loss_callback: Callable[[nn.Module, T.Tensor, T.Tensor], T.Tensor],
            log_dir: Optional[str] = None,
    ):
        # self._schedular = optim.lr_scheduler.StepLR(op,
        #                                             lr_decay_freq,
        #                                             gamma=lr_decay_factor)
        self._writer = writer = SummaryWriter(logdir=log_dir)
        self._global_step = 0
        self._loss_callback = loss_callback
        schedular = self._schedular

        for cur_epoch in tqdm(range(epoch)):
            print('Epoch: {}'.format(cur_epoch))
            self.train_loop(cur_epoch)
            self.valid_loop(cur_epoch)
            schedular.step()  # type: ignore

        writer.close()

    def update_hit_counts(self, users_index: T.Tensor, pos_preds: T.Tensor,
                          neg_preds: T.Tensor) -> None:

        users_index = users_index.to(T.device('cpu'))
        pos_preds = pos_preds.to(T.device('cpu'))
        neg_preds = neg_preds.to(T.device('cpu'))

        binary_ranks = (pos_preds - neg_preds) > 0
        binary_ranks = binary_ranks.to(T.float)
        users, user_counts = T.unique(users_index, return_counts=True)
        user_counts = user_counts.to(T.float)
        self.hit_per_user.scatter_add_(0, users_index, binary_ranks)
        self.user_counts[users] += user_counts

    def compute_auc(self) -> T.Tensor:
        rate = self.hit_per_user / self.user_counts
        rate[T.isnan(rate)] = 0
        rate[T.isinf(rate)] = 0
        auc = T.mean(rate)
        return auc

    def train_loop(self, epoch: int) -> None:

        dl = self.dl_dict['train']
        op = self._op
        schedular = self._schedular
        loss_callback = self._loss_callback
        writer = self._writer
        global_step = self._global_step
        self.hit_per_user.zero_()
        self.user_counts.zero_()
        loss = 0.0

        for step, (user_index, pos_batch, neg_batch) in enumerate(dl):
            op.zero_grad()

            pos_preds, neg_preds = self.model(pos_batch, neg_batch)
            bprloss = loss_callback(self.model, pos_preds, neg_preds)
            bprloss.backward()
            op.step()

            cur_loss = bprloss.item()
            loss += cur_loss

            writer.add_scalar('loss', cur_loss, global_step=global_step)
            print("Epoch {} step {}: training loss: {}".format(
                epoch, global_step, cur_loss))
            global_step += 1
            with T.no_grad():
                self.update_hit_counts(user_index, pos_preds, neg_preds)

        with T.no_grad():
            self.log_info(epoch, step + 1, loss, loop_type='train')

    def valid_loop(self, epoch: int) -> None:
        with T.no_grad():
            dl = self.dl_dict['valid']
            loss_callback = self._loss_callback
            writer = self._writer
            self.hit_per_user.zero_()
            self.user_counts.zero_()
            loss = 0.0

            for step, (user_index, pos_batch, neg_batch) in enumerate(dl):

                pos_preds, neg_preds = self.model(pos_batch, neg_batch)
                bprloss = loss_callback(self.model, pos_preds, neg_preds)
                loss += bprloss.item()
                self.update_hit_counts(user_index, pos_preds, neg_preds)

            self.log_info(epoch, step + 1, loss, loop_type='valid')

    def log_info(self, epoch: int, step: int, loss: float,
                 loop_type: str) -> None:
        writer = self._writer

        loss = loss / step
        auc = self.compute_auc().item()
        print("Epoch {e}: {t} loss {l}, {t} accuarcy {a}".format(e=epoch,
                                                                 t=loop_type,
                                                                 l=loss,
                                                                 a=auc))
        writer.add_scalar("{}/loss".format(loop_type), loss, global_step=epoch)
        writer.add_scalar("{}/accuarcy".format(loop_type),
                          auc,
                          global_step=epoch)

        if loop_type == 'valid' and auc > self.best_val_auc:
            self.best_val_auc = auc
            self.best_epoch = epoch


def simple_loss(linear_reg: float, emb_reg: float, model: TorchFM,
                pos_preds: T.Tensor, neg_preds: T.Tensor) -> T.Tensor:
    param_linear = model.param_linear
    param_emb = model.param_emb

    l2_term = linear_reg * T.sum(T.pow(param_linear, 2))
    l2_term += emb_reg * T.sum(T.pow(param_emb, 2))

    bprloss = T.sum(T.log(1e-10 + T.sigmoid(pos_preds - neg_preds))) - l2_term
    bprloss = -1 * bprloss
    return bprloss


def trans_loss(linear_reg: float, emb_reg: float, trans_reg: float,
               model: TorchTransFM, pos_preds: T.Tensor,
               neg_preds: T.Tensor) -> T.Tensor:
    param_linear = model.param_linear
    param_emb = model.param_linear
    param_trans = model.param_trans

    l2_term = linear_reg * T.sum(T.pow(param_linear, 2))
    l2_term += emb_reg * T.sum(T.pow(param_emb, 2))
    l2_term += trans_reg * T.sum(T.pow(param_trans, 2))

    bprloss = T.sum(T.log(1e-10 + T.sigmoid(pos_preds - neg_preds))) - l2_term
    bprloss = -1 * bprloss
    return bprloss