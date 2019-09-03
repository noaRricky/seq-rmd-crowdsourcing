from pathlib import Path
from typing import List, Dict

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch as T
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from utils import build_logger

logger = build_logger()


class MovelenDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx]


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

        user_one_hot = sp.identity(user_counts.size).tocsr()
        item_one_hot = sp.identity(item_counts.size).tocsr()

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
        self.feat_dim = user_counts.size + 2 * item_counts.size
        self._batch_size = 32
        self._shuffle = False
        self._num_workers = 0
        self._user_one_hot = user_one_hot
        self._item_one_hot = item_one_hot

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
        ds = MovelenDataset(df)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=data_collate,
                        num_workers=num_workers)

        return dl

    def _data_collate(self, batch: List[np.ndarray]):
        device = self._device

        df_batch = pd.DataFrame(batch)
        user_vector = self._user_one_hot[df_batch['user_id']]
        item_vector = self._item_one_hot[df_batch['item_id']]
        prev_vector = self._item_one_hot[df_batch['prev_item_id']]
        neg_vector = self._item_one_hot[df_batch['neg_item_id']]

        pos_feats_matrix = sp.hstack([user_vector, prev_vector, item_vector])
        neg_feats_matrix = sp.hstack([user_vector, prev_vector, neg_vector])
        tensor_size = T.Size(pos_feats_matrix.shape)

        user_tensor = T.tensor(df_batch['user_id'].values, dtype=T.long)

        pos_index_array: np.ndarray = np.vstack(pos_feats_matrix.nonzero())
        pos_index = T.tensor(pos_index_array.tolist())
        pos_value = T.tensor(pos_feats_matrix.data, dtype=T.double)
        pos_tensor = T.sparse_coo_tensor(pos_index,
                                         pos_value,
                                         size=tensor_size)

        neg_index_array: np.ndarray = np.vstack(neg_feats_matrix.nonzero())
        neg_index = T.tensor(neg_index_array.tolist())
        neg_value = T.tensor(neg_feats_matrix.data, dtype=T.double)
        neg_tensor = T.sparse_coo_tensor(neg_index,
                                         neg_value,
                                         size=tensor_size)

        user_tensor = user_tensor.to(device)
        pos_tensor = pos_tensor.to(device)
        neg_tensor = neg_tensor.to(device)

        return user_tensor, pos_tensor, neg_tensor
