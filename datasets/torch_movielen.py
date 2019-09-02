from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch as T
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from utils import build_logger

logger = build_logger()


class MovelenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pos_names: List[str],
                 neg_names: List[str]):
        self.pos_names = pos_names
        self.neg_names = neg_names
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        pos_data = self.df[self.pos_names].iloc[idx].values
        neg_data = self.df[self.neg_names].iloc[idx].values
        return pos_data, neg_data


class TorchMovielen10k:
    def __init__(self, file_path: Path, user_min: int = 4, item_min: int = 4):
        df = pd.read_csv(file_path,
                         header=None,
                         sep='\t',
                         names=['user_id', 'item_id', 'rating', 'time'],
                         dtype={
                             'user_id': 'int32',
                             'item_id': 'int32',
                             'rating': 'int32',
                             'time': 'int32'
                         })

        logger.info("Read dataset in {}".format(file_path))
        user_counts = df.user_id.value_counts()
        item_counts = df.item_id.value_counts()

        logger.info("Original user size: {}".format(user_counts.size))
        logger.info("Original item size: {}".format(item_counts.size))

        # get user and tiem category info
        user_counts = user_counts[user_counts >= user_min]
        item_counts = item_counts[item_counts >= item_min]

        logger.info("Filter user size: {}".format(user_counts.size))
        logger.info("Filter item size: {}".format(item_counts.size))

        # remove sparse item
        df = df[df.user_id.isin(user_counts.index)]
        df = df[df.item_id.isin(item_counts.index)]

        # Add previous item
        df['prev_item_id'] = df.item_id
        df['prev_item_id'] = df['prev_item_id'].shift(
            periods=1).fillna(0).astype(np.int32)

        # Add negtive item
        df['neg_item_id'] = df.item_id.sample(df.shape[0]).values

        # split train and test ddataframe
        df = df.sort_values(by=['time'])
        duplicate_mask = df.duplicated(subset=['user_id'], keep='last')
        remain_df = df[duplicate_mask]
        test_df = df[~duplicate_mask]
        duplicate_mask = remain_df.duplicated(subset=['user_id'], keep='last')
        train_df = remain_df[duplicate_mask]
        valid_df = remain_df[~duplicate_mask]

        # Set first item non for each user
        train_df.sort_values(by=['user_id'])
        first_mask = ~train_df.duplicated(subset=['user_id'], keep='first')
        train_df['prev_item_id'][first_mask] = -1

        # encode feature
        cat_names = ['user_id', 'item_id', 'prev_item_id', 'neg_item_id']
        ordinal_encoder = OrdinalEncoder(categories='auto', dtype='int32')
        ordinal_encoder.fit(train_df[cat_names])

        data = ordinal_encoder.transform(train_df[cat_names])
        train_df[cat_names] = data

        data = ordinal_encoder.transform(valid_df[cat_names])
        valid_df[cat_names] = data

        data = ordinal_encoder.transform(test_df[cat_names])
        test_df[cat_names] = data

        # set train, valid, test
        self.cat_dict: Dict[str, np.ndarray] = {
            name: cat_array
            for name, cat_array in zip(cat_names, ordinal_encoder.categories_)
        }
        self.pos_cat_names: List[str] = [
            'user_id',
            'item_id',
            'prev_item_id',
        ]
        self.neg_cat_names: List[str] = [
            'user_id',
            'neg_item_id',
            'prev_item_id',
        ]
        self.df_dict: Dict[str, pd.DataFrame] = {
            'train': train_df[cat_names],
            'valid': valid_df[cat_names],
            'test': test_df[cat_names]
        }
        self._batch_size = 32
        self._device = T.device('cpu')
        self._shuffle = False
        self._num_workers = 0

    def batch(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle

    def device(self, device: T.device) -> None:
        self._device = device

    def workers(self, num_workers: int = 0) -> None:
        self._num_workers = num_workers

    def get_dataloader(
            self,
            dataset_type: str,
    ) -> DataLoader:
        assert dataset_type in self.df_dict, "Don't contain dataset type"

        shuffle = self._shuffle
        num_workers = self._num_workers
        batch_size = self._batch_size
        data_collate = self._data_collate

        df = self.df_dict[dataset_type]
        ds = MovelenDataset(df, self.pos_cat_names, self.neg_cat_names)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=data_collate,
                        num_workers=num_workers)

        return dl

    def _data_collate(self, batch: List[np.ndarray]):
        pos_batch, neg_batch = zip(*batch)
        pos_tensor = T.tensor(pos_batch, dtype=T.long, device=self._device)
        neg_tensor = T.tensor(neg_batch, dtype=T.long, device=self._device)
        return pos_tensor, neg_tensor
