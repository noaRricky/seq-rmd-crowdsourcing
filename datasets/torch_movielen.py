from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import Dataset

from utils import build_logger

logger = build_logger()


class TorchMovielen10k(Dataset):
    def __init__(self,
                 file_path: Path,
                 dataset_type: str = 'train',
                 user_min: int = 4,
                 item_min=4):
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

        logger.info("Filter user size:", user_counts.size)
        logger.info("Filter item size:", item_counts.size)

        # remove sparse item
        df = df[df.user_id.isin(user_counts.index)]
        df = df[df.item_id.isin(item_counts.index)]

        # Add previous item
        df['prev_item_id'] = df.item_id
        data = df['prev_item_id'].values
        data = np.roll(data, 1)
        df['prev_item_id'].update(pd.Series(data))

        # split train and test ddataframe
        df = df.sort_values(by=['time'])
        duplicate_mask = df.duplicated(subset=['user_id'], keep='last')
        remain_df = df[duplicate_mask]
        test_df = df[~duplicate_mask]
        duplicate_mask = remain_df.duplicated(subset=['user_id'], keep='last')
        train_df = remain_df[duplicate_mask]
        valid_df = remain_df[~duplicate_mask]

        # encode feature
        cat_names = ['user_id', 'item_id', 'prev_item_id']
        ordinal_encoder = OrdinalEncoder(categories='auto')

        encoded_train_data = ordinal_encoder.fit_transform(train_df[cat_names])
        encoded_train_df = pd.DataFrame(data=encoded_train_data,
                                        columns=cat_names,
                                        dtype='int32')
        train_df.update(encoded_train_df)
        encoded_valid_data = ordinal_encoder.transform(valid_df[cat_names])
        encoded_valid_df = pd.DataFrame(data=encoded_valid_data,
                                        columns=cat_names,
                                        dtype='int32')
        valid_df.update(encoded_valid_df)
        encoded_test_data = ordinal_encoder.transform(test_df[cat_names])
        encoded_test_df = pd.DataFrame(data=encoded_test_data,
                                       columns=cat_names,
                                       dtype='int32')
        test_df.update(encoded_test_df)

        # Set first item non for each user
        first_mask = ~train_df.duplicated(subset=['user_id'], keep='first')
        train_df['prev_item_id'][first_mask] = -1

        # set train, valid, test
        self.dataset_type = dataset_type
        self.cat_names = cat_names
        self.cat_dict = {
            name: cat_array
            for name, cat_array in (cat_names, ordinal_encoder.categories_)
        }
        self.train_df = train_df[cat_names]
        self.valid_df = valid_df[cat_names]
        self.test_df = test_df[cat_names]

    def __len__(self):
        if self.dataset_type == 'train':
            return self.train_df.shape[0]
        elif self.dataset_type == 'valid':
            return self.valid_df.shape[0]
        elif self.dataset_type == 'test':
            return self.test_df.shape[0]

    def __getitem__(self, idx):
        return self.train_df.iloc[idx].values
