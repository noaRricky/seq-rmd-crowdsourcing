from typing import List, Dict

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class TorchFM(nn.Module):
    def __init__(
            self,
            cat_dict: Dict[str, np.ndarray],
    ):
        self.cat_dict = cat_dict
        self.emb_layer = nn.ModuleList(
            [nn.Embedding(cat_dict[key].size, 1) for key in cat_dict])

    def forward(self, data_batch):
        x = [data_batch[:, i] for i, emb in enumerate(self.emb_layer)]
        x = T.cat(x, dim=1)
