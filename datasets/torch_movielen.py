from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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
    def __init__(self,
                 data_path: Path,
                 item_path: Optional[Path] = None,
                 user_min: int = 4,
                 item_min: int = 4):
        self._read_data_csv(data_path, user_min, item_min)
        if item_path:
            self._read_item_csv(item_path)

        # self.feat_dim = user_counts.size + 2 * item_counts.size
        # self.user_size = user_counts.size
        self._data_path = data_path
        self._item_path = item_path
        self._batch_size = 32
        self._shuffle = False
        self._num_workers = 0
        self._device = T.device('cpu')
        self.feat_dim = self.user_size + 2 * self.item_size
        if item_path:
            self.feat_dim += 2 * self._item_seq_size

        # self._user_one_hot = user_one_hot
        # self._item_one_hot = item_one_hot

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

    def _read_data_csv(self,
                       data_path: Path,
                       user_min: int = 4,
                       item_min: int = 4):
        df = pd.read_csv(data_path,
                         header=None,
                         sep='\t',
                         names=['user_id', 'item_id', 'rating', 'time'],
                         dtype={
                             'user_id': 'int32',
                             'item_id': 'int32',
                             'rating': 'int32',
                             'time': 'int32'
                         })

        logger.info("Read dataset in {}".format(data_path))
        user_counts = df.user_id.value_counts()
        item_counts = df.item_id.value_counts()

        logger.info("Original user size: {}".format(user_counts.size))
        logger.info("Original item size: {}".format(item_counts.size))

        # get user and item category info
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

        cat_names = ['user_id', 'item_id', 'prev_item_id', 'neg_item_id']

        # Build encoder
        user_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        item_encoder = OneHotEncoder(categories='auto',
                                     handle_unknown='ignore')
        user_encoder.fit(train_df[['user_id']])
        item_encoder.fit(train_df[['item_id']])

        self.df_dict: Dict[str, pd.DataFrame] = {
            'train': train_df[cat_names],
            'valid': valid_df[cat_names],
            'test': test_df[cat_names]
        }
        self._user_encoder = user_encoder
        self._item_encoder = item_encoder
        self.user_size = user_counts.size
        self.item_size = item_counts.size

    def _read_item_csv(self, item_file_path: Path) -> None:
        base_cols = [
            'movie_id', 'movie_title', 'release_date', 'video_release_date',
            'imdb_url'
        ]
        movie_cat_cols = [
            'unkown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            'Western'
        ]

        item_df = pd.read_csv(item_file_path,
                              header=None,
                              index_col=False,
                              sep='|',
                              names=base_cols + movie_cat_cols)
        item_df = item_df[['movie_id'] + movie_cat_cols]
        item_df[movie_cat_cols] = item_df[movie_cat_cols].astype('float64')
        item_df = item_df.set_index('movie_id')

        item_seq_size = len(movie_cat_cols)
        item_df.loc[-1] = np.zeros(item_seq_size)

        self._item_df = item_df
        self._item_seq_size = item_seq_size

    def _data_collate(self, batch: List[np.ndarray]):
        device = self._device

        df_batch = pd.DataFrame(batch)

        # get index lists
        user_vector: sp.csr_matrix = self._user_encoder.transform(
            df_batch[['user_id']])
        item_vector: sp.csr_matrix = self._item_encoder.transform(
            df_batch[['item_id']])
        prev_vector: sp.csr_matrix = self._item_encoder.transform(
            df_batch[['prev_item_id']])
        neg_vector: sp.csr_matrix = self._item_encoder.transform(
            df_batch[['neg_item_id']])

        pos_feats_matrix = sp.hstack([user_vector, prev_vector, item_vector])
        neg_feats_matrix = sp.hstack([user_vector, prev_vector, neg_vector])

        if self._item_path:
            item_seq_vector = sp.csr_matrix(
                self._item_df.loc[df_batch['item_id']])
            prev_seq_vector = sp.csr_matrix(
                self._item_df.loc[df_batch['prev_item_id']])
            neg_seq_vector = sp.csr_matrix(
                self._item_df.loc[df_batch['neg_item_id']])

            pos_feats_matrix = sp.hstack(
                [pos_feats_matrix, prev_seq_vector, item_seq_vector])
            neg_feats_matrix = sp.hstack(
                [neg_feats_matrix, prev_seq_vector, neg_seq_vector])

        user_tensor = T.tensor(user_vector.indices, dtype=T.long)
        pos_tensor = self._build_feat_tensor(pos_feats_matrix)
        neg_tensor = self._build_feat_tensor(neg_feats_matrix)

        user_tensor = user_tensor.to(device)
        pos_tensor = pos_tensor.to(device)
        neg_tensor = neg_tensor.to(device)

        return user_tensor, pos_tensor, neg_tensor

    def _build_feat_tensor(self, feat_matrix: sp.coo_matrix):
        feat_index_list = feat_matrix.nonzero()
        feat_index_array: np.ndarray = np.vstack(feat_index_list)

        feat_index = T.tensor(feat_index_array.tolist())
        feat_value = T.tensor(feat_matrix.data, dtype=T.double)
        feat_tensor = T.sparse_coo_tensor(feat_index_list,
                                          feat_value,
                                          size=feat_matrix.shape)

        return feat_tensor


if __name__ == "__main__":
    data_path = Path("./inputs/ml-100k/u.data")
    item_path = Path("./inputs/ml-100k/u.item")
    databunch = TorchMovielen10k(data_path, item_path)

    train_dl = databunch.get_dataloader(dataset_type='train')
    train_it = iter(train_dl)

    print(train_it.next())  # type: ignore
