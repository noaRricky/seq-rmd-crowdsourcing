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


class TorchPrmeFM(TorchFM):
    def forward(self, pos_feats, neg_feats):
        param_linear = self.param_linear
        param_factor = self.param_factor

        # Linear Term
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interactive term
        var_emb_product = T.sum(T.pow(param_factor, 2), dim=1, keepdim=True)

        # Common term positive
        pos_feats_sum = T.sum(pos_feats, dim=1, keepdim=True)
        pos_emb_mul = T.mm(pos_feats, param_factor)

        # Common term negtive
        neg_feats_sum = T.sum(neg_feats, dim=1, keepdim=True)
        neg_emb_mul = T.mm(neg_feats, param_factor)

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


class TorchHrmFM(TorchFM):
    def forward(self, pos_feats, neg_feats):
        param_linear = self.param_linear
        param_factor = self.param_factor

        # Linear term
        pos_linear = T.mm(pos_feats, param_linear)
        neg_linear = T.mm(neg_feats, param_linear)

        # Interaction terms
        # First define common terms that are used by future calculations
        # Common terms
        var_emb_product = T.sum(T.pow(param_factor, 2), dim=1, keepdim=True)

        # Common term positive
        pos_feats_sum = T.sum(pos_feats, dim=1, keepdim=True)
        pos_emb_mul = T.mm(pos_feats, param_factor)

        # Common terms negative
        neg_feats_sum = T.sum(neg_feats, dim=1, keepdim=True)
        neg_emb_mul = T.mm(neg_feats, param_factor)

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


class TorchTransFM(TorchFM):
    def forward(self, pos_feats, neg_feats):
        raise NotImplementedError
