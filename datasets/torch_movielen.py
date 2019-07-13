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


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_type: str):
        self.dataset_type = dataset_type
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx].values


class TorchMovielen10k:
    def __init__(self, file_path: Path, user_min: int = 4, item_min=4):
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
        df['prev_item_id'] = df.prev_item_id.shift(periods=1)

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
        self.cat_dict = {
            name: cat_array
            for name, cat_array in zip(cat_names, ordinal_encoder.categories_)
        }
        self.df_dict = {
            'train': train_df[cat_names],
            'valid': valid_df[cat_names],
            'test': test_df[cat_names]
        }

    def get_dataloader(self,
                       dataset_type: str,
                       device: T.device = T.device('cpu'),
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 4) -> DataLoader:
        assert dataset_type in self.df_dict, "Don't contain dataset type"

        self.device = device
        ds = TabularDataset(self.df_dict[dataset_type], dataset_type)
        return DataLoader(ds,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=self.data_collate,
                          num_workers=num_workers)

    def data_collate(self, batch: List[np.ndarray]):
        return T.tensor(batch, dtype=T.long, device=self.device)


if __name__ == "__main__":
    file_path = Path('inputs/ml-100k/u.data')

    movelen = TorchMovielen10k(file_path, user_min=4, item_min=4)
    train_dl = movelen.get_dataloader('train')

    for batch in train_dl:
        print(batch)
        break
