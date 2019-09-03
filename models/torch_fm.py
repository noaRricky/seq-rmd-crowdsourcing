from typing import List, Dict

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class TorchFM(nn.Module):
    def __init__(self, feature_dim: int, num_dims: int, init_mean: int):
        super(TorchFM, self).__init__()

        var_linear = T.rand(feature_dim, 1,
                            dtype=T.double) * init_mean * 2 - init_mean
        var_factor = T.rand(feature_dim, num_dims,
                            dtype=T.double) * init_mean * 2 - init_mean
        self.param_linear = nn.Parameter(var_linear)
        self.param_factor = nn.Parameter(var_factor)

    def forward(self, pos_feats, neg_feats):
        param_linear = self.param_linear
        param_factor = self.param_factor

        # Linear terms
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interactive terms
        pos_emb_mul = T.mm(pos_feats, param_factor)
        term_1_pos = T.pow(T.sum(pos_emb_mul, dim=1, keepdim=True), 2)
        term_2_pos = T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        pos_preds = pos_linear + 0.5 * (term_1_pos - term_2_pos)

        neg_emb_mul = T.mm(neg_feats, param_factor)
        term_1_neg = T.pow(T.sum(neg_emb_mul, dim=1, keepdim=True), 2)
        term_2_neg = T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)
        # Shape: (batch_size, 1)
        neg_preds = neg_linear + 0.5 * (term_2_neg - term_2_neg)

        # Shape: (batch_size)
        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds
