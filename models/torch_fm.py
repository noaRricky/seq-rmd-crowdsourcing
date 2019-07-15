from typing import List, Dict, Callable, Optional

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
from datasets.torch_movielen import TorchMovielen10k

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
    def __init__(self,
                 model: TorchFM,
                 optimzer: Callable,
                 databunch: TorchMovielen10k,
                 device: T.device = T.device('cpu')) -> None:

        self.model = model.to(device)
        self.train_dl = databunch.get_dataloader('train', device=device)
        self.valid_dl = databunch.get_dataloader('valid', device=device)
        self.test_dl = databunch.get_dataloader('test', device=device)
        self.optimizer = optimzer

        # Genrate accuracy per user to compute
        users = databunch.cat_dict['user_id']
        self.hit_per_user: T.Tensor = T.zeros(users.size)
        self.user_counts: T.Tensor = T.zeros(users.size)

    def fit(
            self,
            epoch: int,
            lr: float,
            log_dir: Optional[str] = None,
            lr_decay_factor: float = 1,
            lr_decay_freq: int = 1000,
            linear_reg: float = 0.0,
            factor_reg: float = 0.0,
    ):
        op = self.optimizer(self.model.parameters(), lr=lr)
        schedular = optim.lr_scheduler.StepLR(op,
                                              lr_decay_freq,
                                              gamma=lr_decay_factor)
        writer = SummaryWriter(logdir=log_dir)
        global_step = 0.0

        for cur_epoch in tqdm(range(epoch)):
            print('Epoch: {}'.format(cur_epoch))

            # Train loop
            self.hit_per_user.zero_()
            train_loss = 0.0
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

                cur_loss = bprloss.item()
                train_loss += cur_loss

                writer.add_scalar('loss', cur_loss, global_step=global_step)
                global_step += 1

                with T.no_grad():
                    self.update_hit_counts(pos_batch, pos_preds, neg_preds)

            train_loss = train_loss / (step + 1)
            auc = self.compute_auc().item()

            # Print loss and accuarcy
            print("Epoch {}: train loss {}, train accuarcy {}".format(
                cur_epoch, train_loss, auc))
            writer.add_scalar('train/loss', train_loss, global_step=cur_epoch)
            writer.add_scalar('train/accuarcy', auc, global_step=cur_epoch)
            # Validation loop
            with T.no_grad():

                self.hit_per_user.zero_()
                valid_loss = 0.0
                for step, (pos_batch, neg_batch) in enumerate(self.test_dl):
                    pos_preds, neg_preds = self.model(pos_batch, neg_batch)
                    bprloss = self.criterion(pos_batch,
                                             neg_preds,
                                             linear_reg=linear_reg,
                                             factor_reg=factor_reg)
                    # Compute valid loss and accuarcy for
                    cur_loss = bprloss.item()
                    valid_loss += cur_loss
                    self.update_hit_counts(pos_batch, pos_preds, neg_preds)

                valid_loss = valid_loss / (step + 1)
                valid_auc = self.compute_auc().item()
                # Print loss and accuarcy
                print("Epoch {}: valid loss {}, valid accuarcy {}".format(
                    cur_epoch, valid_loss, valid_auc))
                writer.add_scalar('valid/loss',
                                  valid_loss,
                                  global_step=cur_epoch)
                writer.add_scalar('valid/accuarcy',
                                  valid_auc,
                                  global_step=cur_epoch)

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

    def update_hit_counts(self, pos_batch: T.Tensor, pos_preds: T.Tensor,
                          neg_preds: T.Tensor) -> None:

        binary_ranks = (pos_preds - neg_preds) > 0
        binary_ranks = binary_ranks.to(T.float)
        users_index = pos_batch[:, 0]
        users, user_counts = T.unique(users_index, return_counts=True)
        user_counts = user_counts.to(T.float)
        self.hit_per_user.scatter_add_(0, users_index, binary_ranks)
        self.user_counts.scatter_add_(0, users_index, user_counts)

    def compute_auc(self) -> T.Tensor:
        auc = T.mean(self.hit_per_user / self.user_counts)
        return auc
