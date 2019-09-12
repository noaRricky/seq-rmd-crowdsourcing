from typing import List, Dict

import numpy as np
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

    def get_dataloader(self, ds_type: str) -> DataLoader:
        assert ds_type in self._df_dict, "Don't contain dataset type"

        shuffle = self.shuffle
        num_workers = self.num_workers
        batch_size = self.batch_size
        data_collate = self._data_collate

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
                  device: T.device = T.device('cpu')) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device

    def _data_collate(self, batch: List[np.ndarray]):
        raise NotImplementedError
