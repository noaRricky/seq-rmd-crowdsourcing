from typing import List, Dict

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class TorchFM(nn.Module):
    def __init__(self, feature_dim: int, num_dim: int, init_mean: int):
        super(TorchFM, self).__init__()

        var_linear = T.rand(feature_dim, 1,
                            dtype=T.double) * init_mean * 2 - init_mean
        var_emb = T.rand(feature_dim, num_dim,
                         dtype=T.double) * init_mean * 2 - init_mean

        self._feature_dim = feature_dim
        self._num_dim = num_dim
        self._init_mean = init_mean
        self.param_linear = nn.Parameter(var_linear)
        self.param_emb = nn.Parameter(var_emb)

    def forward(self, pos_feats, neg_feats):
        param_linear = self.param_linear
        param_emb = self.param_emb

        # Linear terms
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interactive terms
        pos_emb_mul = T.mm(pos_feats, param_emb)
        term_1_pos = T.pow(T.sum(pos_emb_mul, dim=1, keepdim=True), 2)
        term_2_pos = T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        pos_preds = pos_linear + 0.5 * (term_1_pos - term_2_pos)

        neg_emb_mul = T.mm(neg_feats, param_emb)
        term_1_neg = T.pow(T.sum(neg_emb_mul, dim=1, keepdim=True), 2)
        term_2_neg = T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)
        # Shape: (batch_size, 1)
        neg_preds = neg_linear + 0.5 * (term_2_neg - term_2_neg)

        # Shape: (batch_size)
        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds


class TorchPrmeFM(nn.Module):
    def __init__(self, feature_dim: int, num_dim: int, init_mean: int):
        super(TorchPrmeFM, self).__init__()

        var_linear = T.rand(feature_dim, 1,
                            dtype=T.double) * init_mean * 2 - init_mean
        var_emb = T.rand(feature_dim, num_dim,
                         dtype=T.double) * init_mean * 2 - init_mean

        self._feature_dim = feature_dim
        self._num_dim = num_dim
        self._init_mean = init_mean
        self.param_linear = nn.Parameter(var_linear)
        self.param_emb = nn.Parameter(var_emb)

    def forward(self, pos_feats, neg_feats):
        feature_dim = self._feature_dim
        param_linear = self.param_linear
        param_emb = self.param_emb

        # Linear Term
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interactive term
        var_emb_product = T.sum(T.pow(param_emb, 2), dim=1, keepdim=True)

        # generate sum value
        var_sum_op = T.ones((feature_dim, 1),
                            dtype=T.double,
                            device=pos_feats.device)

        # Common term positive
        pos_feats_sum = T.mm(pos_feats, var_sum_op)
        pos_emb_mul = T.mm(pos_feats, param_emb)

        # Common term negtive
        neg_feats_sum = T.mm(neg_feats, var_sum_op)
        neg_emb_mul = T.mm(neg_feats, param_emb)

        # Term 1 pos
        prod_term_pos = T.mm(pos_feats, var_emb_product)
        term_1_pos = prod_term_pos * pos_feats_sum

        # Term 1 neg
        prod_term_neg = T.mm(neg_feats, var_emb_product)
        term_1_neg = prod_term_neg * neg_feats_sum

        # Term 2
        term_2_pos = 2 * T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        term_2_neg = 2 * T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)

        # Term 3
        term_3_pos = term_1_pos
        term_3_neg = term_1_neg

        # Prediction
        pos_preds = pos_linear + 0.5 * (term_1_pos - term_2_pos + term_3_pos)
        neg_preds = neg_linear + 0.5 * (term_1_neg - term_2_neg + term_3_neg)

        # Shape (batch_size)
        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds


class TorchHrmFM(nn.Module):
    def __init__(self, feature_dim: int, num_dim: int, init_mean: int):
        super(TorchHrmFM, self).__init__()

        var_linear = T.rand(feature_dim, 1,
                            dtype=T.double) * init_mean * 2 - init_mean
        var_emb = T.rand(feature_dim, num_dim,
                         dtype=T.double) * init_mean * 2 - init_mean

        self._feature_dim = feature_dim
        self._num_dim = num_dim
        self._init_mean = init_mean
        self.param_linear = nn.Parameter(var_linear)
        self.param_emb = nn.Parameter(var_emb)

    def forward(self, pos_feats, neg_feats):
        feature_dim = self._feature_dim
        param_linear = self.param_linear
        param_emb = self.param_emb

        # Linear term
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interaction terms
        # First define common terms that are used by future calculations
        # Common terms
        var_emb_product = T.sum(T.pow(param_emb, 2), dim=1, keepdim=True)
        var_sum_op = T.ones((feature_dim, 1),
                            dtype=T.double,
                            device=pos_feats.device)

        # Common term positive
        pos_feats_sum = T.mm(pos_feats, var_sum_op)
        pos_emb_mul = T.mm(pos_feats, param_emb)

        # Common terms negative
        neg_feats_sum = T.mm(neg_feats, var_sum_op)
        neg_emb_mul = T.mm(neg_feats, param_emb)

        # Term 1 pos
        prod_term_pos = T.mm(pos_feats, var_emb_product)
        term_1_pos = prod_term_pos * pos_feats_sum

        # Term 1 neg
        prod_term_neg = T.mm(neg_feats, var_emb_product)
        term_1_neg = prod_term_neg * neg_feats_sum

        # Term 2
        term_2_pos = T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        term_2_neg = T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)

        # Diag term
        diag_term_pos = T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        diag_term_neg = T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)

        # Predictions
        pos_preds = pos_linear + 0.5 * (term_1_pos +
                                        term_2_pos) - diag_term_pos
        neg_preds = neg_linear + 0.5 * (term_1_neg +
                                        term_2_neg) - diag_term_neg

        # Shape (batch_size)
        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds


class TorchTransFM(nn.Module):
    def __init__(self, feature_dim: int, num_dim: int, init_mean: int):
        super(TorchTransFM, self).__init__()

        var_linear = T.rand(feature_dim, 1,
                            dtype=T.double) * init_mean * 2 - init_mean
        var_emb = T.rand(feature_dim, num_dim,
                         dtype=T.double) * init_mean * 2 - init_mean

        self._feature_dim = feature_dim
        self._num_dim = num_dim
        self._init_mean = init_mean
        self.param_linear = nn.Parameter(var_linear)
        self.param_emb = nn.Parameter(var_emb)

        var_trans = T.rand(feature_dim, num_dim,
                           dtype=T.double) * init_mean * 2 - init_mean
        self.param_trans = nn.Parameter(var_trans)

    def forward(self, pos_feats, neg_feats):
        feature_dim = self._feature_dim
        param_linear = self.param_linear
        param_emb = self.param_emb
        param_trans = self.param_trans

        # Linear term
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # print(pos_linear.shape)
        # print(neg_linear.shape)

        # Interaction terms
        # First define common terms that are used by future calculations
        # Common terms
        var_sum_op = T.ones((feature_dim, 1),
                            dtype=T.double,
                            device=pos_feats.device)
        var_emb_product = T.sum(T.pow(param_emb, 2), dim=1, keepdim=True)
        var_trans_product = T.sum(T.pow(param_trans, 2), dim=1, keepdim=True)
        var_emb_trans_product = T.sum(param_emb * param_trans,
                                      dim=1,
                                      keepdim=True)

        # Common term positive
        pos_feats_sum = T.mm(pos_feats, var_sum_op)
        pos_emb_mul = T.mm(pos_feats, param_emb)
        pos_trans_mul = T.mm(pos_feats, param_trans)

        # Common terms negative
        neg_feats_sum = T.mm(neg_feats, var_sum_op)
        neg_emb_mul = T.mm(neg_feats, param_emb)
        neg_trans_mul = T.mm(neg_feats, param_trans)

        # Term 1 pos
        prod_term_pos = T.mm(pos_feats, var_emb_product)
        term_1_pos = prod_term_pos * pos_feats_sum

        # Term 1 neg
        prod_term_neg = T.mm(neg_feats, var_emb_product)
        term_1_neg = prod_term_neg * neg_feats_sum

        # Term 2 pos
        prod_term_pos = T.mm(pos_feats, var_trans_product)
        term_2_pos = prod_term_pos * pos_feats_sum

        # Term 2 neg
        prod_term_neg = T.mm(neg_feats, var_trans_product)
        term_2_neg = prod_term_neg * pos_feats_sum

        # Term 3
        term_3_pos = term_1_pos
        term_3_neg = term_1_neg

        # Term 4 pos
        prod_term_pos = T.mm(pos_feats, var_emb_trans_product)
        term_4_pos = 2 * prod_term_pos * pos_feats_sum

        # Term 4 neg
        prod_term_neg = T.mm(neg_feats, var_emb_trans_product)
        term_4_neg = 2 * prod_term_neg * neg_feats_sum

        # Term 5
        term_5_pos = 2 * T.sum(T.pow(pos_emb_mul, 2), dim=1, keepdim=True)
        term_5_neg = 2 * T.sum(T.pow(neg_emb_mul, 2), dim=1, keepdim=True)

        # Term 6
        term_6_pos = 2 * T.sum(
            pos_trans_mul * pos_emb_mul, dim=1, keepdim=True)
        term_6_neg = 2 * T.sum(
            neg_trans_mul * neg_emb_mul, dim=1, keepdim=True)

        # Diag term
        diag_term_pos = T.sum(T.pow(pos_trans_mul, 2), dim=1, keepdim=True)
        diag_term_neg = T.sum(T.pow(neg_trans_mul, 2), dim=1, keepdim=True)

        # Predictions
        pos_preds = pos_linear + 0.5 * (term_1_pos + term_2_pos + term_3_pos +
                                        term_4_pos - term_5_pos -
                                        term_6_pos) - 0.5 * diag_term_pos
        neg_preds = neg_linear + 0.5 * (term_1_neg + term_2_neg + term_3_neg +
                                        term_4_neg - term_5_neg -
                                        term_6_neg) - 0.5 * diag_term_neg

        # Shape (batch_size)
        pos_preds = pos_preds.squeeze()
        neg_preds = neg_preds.squeeze()

        return pos_preds, neg_preds
