from typing import Callable, List, Dict, Optional

import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch as T
from torch.utils.data import Dataset, DataLoader


class DFDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx]


class DataBunch(object):
    def __init__(self):
        self.batch_size = 32
        self.shuffle = False
        self.num_workers = 0
        self.device = T.device('cpu')
        self._df_dict: Dict[str, pd.DataFrame] = {}

    def get_dataloader(self, ds_type: str,
                       collate_fn: str = 'base') -> DataLoader:
        assert ds_type in self._df_dict, "Don't contain dataset type"
        assert collate_fn in ['base', 'seq'], "Don't contain collate function"

        shuffle = self.shuffle
        num_workers = self.num_workers
        batch_size = self.batch_size
        data_collate = self._base_collate
        if collate_fn == 'base':
            data_collate = self._base_collate
        elif collate_fn == 'seq':
            data_collate = self._seq_collate

        df = self._df_dict[ds_type]
        ds = DFDataset(df)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=data_collate,
                        num_workers=num_workers)

        return dl

    def config_db(self,
                  batch_size: int = 32,
                  shuffle: bool = False,
                  num_workers: int = 0,
                  device: T.device = T.device('cpu'),
                  neg_sample: int = 5,
                  collate_fn: str = 'base') -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device
        self.neg_sample = neg_sample
        self.collate_fn = collate_fn

    def _build_feat_tensor(self,
                           feat_matrix: sp.coo_matrix,
                           device: Optional[T.device] = T.device('cpu')):
        feat_index_list = feat_matrix.nonzero()
        feat_index_array: np.ndarray = np.vstack(feat_index_list)

        feat_index = T.tensor(feat_index_array.tolist(),
                              dtype=T.long,
                              device=device)
        feat_value = T.tensor(feat_matrix.data, dtype=T.double, device=device)
        feat_tensor = T.sparse_coo_tensor(feat_index,
                                          feat_value,
                                          size=feat_matrix.shape,
                                          device=device)

        return feat_tensor

    def _base_collate(self, batch: List[np.ndarray]):
        raise NotImplementedError

    def _seq_collate(self, batch: List[np.array]):
        raise NotImplementedError
