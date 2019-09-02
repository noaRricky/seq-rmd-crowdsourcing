from typing import List, Dict

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class TorchHRMFM(nn.Module):
    def __init__(self,
                 cat_dict: Dict[str, np.ndarray],
                 pos_cat_names: List[str],
                 neg_cat_names: List[str],
                 num_dim: int = 10) -> None:
        super(TorchHRMFM, self).__init__()

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

    def forward(self, pos_batch: T.Tensor, neg_batch: T.Tensor) -> T.Tensor:

        # Linear terms
        pos_linear = self.compute_linear_term(pos_batch)
        neg_linear = self.compute_factor_term(neg_batch)

        # Interaction terms
        # First define common terms that are used by future calculations
        # Common terms

        # Common terms positive
        pos_feat

    def compute_linear_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        linear_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_linear_layer)
        ]
        linear = T.sum(T.stack(linear_list), dim=0)

        return linear

    def compute_mul_term(self, batch: T.Tensor) -> T.Tensor:
        batch_size, _ = batch.shape

        emb_list = [
            emb(batch[:, i]) for i, emb in enumerate(self.emb_factor_layer)
        ]
        emb_mul = T.sum(T.stack(emb_list), dim=0)

        return emb_mul
