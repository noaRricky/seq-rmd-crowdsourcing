from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch as T
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from datasets.base import DFDataset, DataBunch


class TorchKaggle(DataBunch):
    def __init__(self, data_path: Path, user_min: Optional[int] = 4) -> None:
        item_df = pd.read_csv(data_path)

        use_cols = [
            'name', 'competitionId', 'deadline', 'rank', 'score',
            'rewardQuantity', 'tier'
        ]
        str_cols = ['name', 'tier']
        # Fresh dataframe
        item_df = item_df[use_cols]
        item_df['deadline'] = pd.to_datetime(item_df['deadline'],
                                             infer_datetime_format=True)
        item_df[str_cols] = item_df[str_cols].astype('str')
        item_df['score'] = item_df['score'].where(item_df['score'] != np.inf,
                                                  0)
        item_df['rewardQuantity'] = item_df['rewardQuantity'].where(
            item_df['rewardQuantity'] != np.inf, 0)
        print(f"Raw dataframe shape {item_df.shape}")
        item_df = item_df.dropna()
        print(f"After drop nan shape: {item_df.shape}")

        # Remove sparse users
        comp_counts = item_df['competitionId'].value_counts()
        name_counts = item_df['name'].value_counts()
        print(f"Original comptition size: {comp_counts.size}")
        print(f"Original competitor size: {name_counts.size}")

        name_counts = name_counts[name_counts >= user_min]
        print(f"Filtered competiter size: {name_counts.size}")

        item_df = item_df[item_df['name'].isin(name_counts.index)]
        print(f"Filtered dataframe shape: {item_df.shape}")

        # Generate competition dataframe for negtive sampling
        comp_cols = ['competitionId', 'rewardQuantity']
        comp_df = item_df[comp_cols]
        comp_df = comp_df.drop_duplicates(subset=['competitionId'],
                                          keep='first')

        # Generate previous columns
        new_cols = ['previousId', 'previousReward']
        pre_cols = ['previousId', 'previousReward', 'rank', 'score', 'tier']
        pre_cont_cols = ['rank', 'score', 'rewardQuantity', 'previousReward']

        item_df = item_df.sort_values(by=['name', 'deadline'])
        item_df[new_cols] = item_df[['competitionId', 'rewardQuantity']]
        item_df[pre_cols] = item_df[pre_cols].shift(periods=1)

        # Mask first competition for each user
        first_mask = item_df.duplicated(subset=['name'], keep='first')
        item_df['previousId'] = item_df['previousId'].fillna(-1).astype(
            'int64').where(first_mask, -1)
        item_df[pre_cont_cols] = item_df[pre_cont_cols].where(first_mask, 0.0)
        item_df['tier'] = item_df['tier'].where(first_mask, 'none')

        # Build up encoder
        item_encoder = OneHotEncoder(categories='auto',
                                     handle_unknown='ignore')
        user_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        cat_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        reward_scaler = MinMaxScaler(feature_range=(0, 1))
        cont_scaler = MinMaxScaler(feature_range=(0, 1))
        cont_names = ['rank', 'score']
        cat_names = ['tier']

        item_encoder.fit(item_df[['competitionId']])
        user_encoder.fit(item_df[['name']])
        cat_encoder.fit(item_df[cat_names])
        cont_scaler.fit(item_df[cont_names])
        reward_scaler.fit(item_df[['rewardQuantity']])

        # Split train, valid, test dataframe
        item_df = item_df.sort_values(by=['name', 'deadline'])
        duplicate_mask = item_df.duplicated(subset=['name'], keep='last')
        remain_df = item_df[duplicate_mask]
        test_df = item_df[~duplicate_mask]
        duplicate_mask = remain_df.duplicated(subset=['name'], keep='last')
        train_df = remain_df[duplicate_mask]
        valid_df = remain_df[~duplicate_mask]

        # count feature dimension
        item_dim = item_encoder.categories_[0].size
        user_dim = user_encoder.categories_[0].size
        cont_dim = len(cont_names)
        cat_dim = sum([c.size for c in cat_encoder.categories_])

        # Configure private attribution
        self._item_encoder = item_encoder
        self._user_encoder = user_encoder
        self._cat_encoder = cat_encoder
        self._cont_scaler = cont_scaler
        self._reward_scaler = reward_scaler
        self._cat_names = cat_names
        self._cont_names = cont_names
        self._df_dict: Dict[str, pd.DataFrame] = {
            'train': train_df,
            'valid': valid_df,
            'test': test_df
        }
        self._comp_df = comp_df

        # Configure public attribution
        self.user_size = user_dim
        self.config_db()
        self.feat_dim = user_dim + 2 * item_dim + cont_dim + cat_dim + 2  # 2 for the reward dimension

    def _data_collate(self, batch: List[pd.Series]):
        device = self.device
        item_encoder = self._item_encoder
        user_encoder = self._user_encoder
        cat_encoder = self._cat_encoder
        cont_scaler = self._cont_scaler
        reward_scaler = self._reward_scaler
        cont_names = self._cont_names
        cat_names = self._cat_names
        comp_df = self._comp_df

        df = pd.DataFrame(batch)
        neg_df = comp_df.sample(n=df.shape[0], replace=True)

        user_matrix: sp.csr_matrix = user_encoder.transform(df[['name']])
        item_matrix = item_encoder.transform(df[['competitionId']])
        prev_matrix = item_encoder.transform(df[['previousId']])
        neg_item_matrix = item_encoder.transform(neg_df[['competitionId']])
        cat_matrix = cat_encoder.transform(df[cat_names])
        cont_matrix = sp.csr_matrix(cont_scaler.transform(df[cont_names]))
        reward_matrix = sp.csr_matrix(
            reward_scaler.transform(df[['rewardQuantity']]))
        prev_reward_matrix = sp.csr_matrix(
            reward_scaler.transform(df[['previousReward']]))
        neg_reward_matrix = sp.csr_matrix(neg_df[['rewardQuantity']])

        pos_matrix = sp.hstack([
            user_matrix, prev_matrix, item_matrix, cat_matrix, cont_matrix,
            prev_reward_matrix, reward_matrix
        ])
        neg_matrix = sp.hstack([
            user_matrix, prev_matrix, neg_item_matrix, cat_matrix, cont_matrix,
            prev_reward_matrix, neg_reward_matrix
        ])

        user_tensor = T.tensor(user_matrix.indices,
                               dtype=T.long,
                               device=device)
        pos_tensor = self._build_feat_tensor(pos_matrix, device=device)
        neg_tensor = self._build_feat_tensor(neg_matrix, device=device)

        return user_tensor, pos_tensor, neg_tensor, None


if __name__ == "__main__":
    item_path = Path("./inputs/kaggle/item.csv")

    databunch = TorchKaggle(item_path)
    train_ld = databunch.get_dataloader(ds_type='train')
    train_it = iter(train_ld)

    user_tensor, pos_tensor, neg_tensor, per_tensor = next(train_it)
    print(per_tensor)
