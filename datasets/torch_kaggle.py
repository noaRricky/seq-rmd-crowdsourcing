from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import torch as T
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from datasets.base import DFDataset, DataBunch


class SeqKaggle(DataBunch):
    def __init__(self, data_path: Path, user_min: int = 4):
        item_df: pd.DataFrame = pd.read_csv(data_path)

        use_cols = [
            'name', 'competitionId', 'deadline', 'rank', 'score',
            'rewardQuantity', 'tier'
        ]
        str_cols = ['name', 'tier']
        # Fresh dataframe
        item_df = item_df[use_cols]
        item_df['period'] = item_df['deadline'].str[:7]
        item_df['deadline'] = pd.to_datetime(item_df['deadline'])
        item_df['period'] = pd.to_datetime(item_df['period'])
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
        comp_cols = ['competitionId', 'rewardQuantity', 'period']
        comp_df = item_df[comp_cols]
        comp_df = comp_df.drop_duplicates(subset=['competitionId'],
                                          keep='first')

        # Transform period column
        bins = pd.date_range(start='2010-3-1', end='2019-12-1', freq='6M')
        comp_df['period'] = pd.cut(comp_df['period'], bins)
        item_df['period'] = pd.cut(item_df['period'], bins)

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
        period_encoder = OrdinalEncoder(categories='auto')
        reward_scaler = MinMaxScaler(feature_range=(0, 1))
        cont_scaler = MinMaxScaler(feature_range=(0, 1))
        cont_names = ['rank', 'score']
        cat_names = ['tier']
        # Fit data to encoder
        item_encoder.fit(item_df[['competitionId']])
        user_encoder.fit(item_df[['name']])
        cat_encoder.fit(item_df[cat_names])
        period_encoder.fit(item_df[['period']])
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
        self._period_encoder = period_encoder
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
        self.feat_dim = user_dim + 2 * item_dim + cont_dim + cat_dim + 2

    def _base_collate(self, batch: List[pd.Series]):
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

    def _seq_collate(self, batch: List[pd.Series]):
        device = self.device
        item_encoder = self._item_encoder
        user_encoder = self._user_encoder
        cat_encoder = self._cat_encoder
        period_encoder = self._period_encoder
        cont_scaler = self._cont_scaler
        reward_scaler = self._reward_scaler
        cont_names = self._cont_names
        cat_names = self._cat_names
        comp_df = self._comp_df
        neg_sample = self.neg_sample

        df = pd.DataFrame(batch)
        # Geneate negative sample
        per_counts = df['period'].value_counts().sort_index()
        per_counts = per_counts * neg_sample
        neg_list = [
            comp_df[comp_df['period'] == per].sample(n=per_counts[per],
                                                     replace=True)
            for per in per_counts.index
        ]
        neg_df = pd.concat(neg_list)

        # Encode perivous and postive feature
        user_matrix: sp.csr_matrix = user_encoder.transform(
            df[['name']].values.repeat(neg_sample, axis=0))
        item_matrix = item_encoder.transform(df[['competitionId'
                                                 ]].values.repeat(neg_sample,
                                                                  axis=0))
        prev_matrix = item_encoder.transform(df[['previousId'
                                                 ]].values.repeat(neg_sample,
                                                                  axis=0))
        cat_matrix = cat_encoder.transform(df[cat_names].values.repeat(
            neg_sample, axis=0))
        cont_matrix = sp.csr_matrix(
            cont_scaler.transform(df[cont_names].values.repeat(neg_sample,
                                                               axis=0)))
        reward_matrix = sp.csr_matrix(
            reward_scaler.transform(df[['rewardQuantity'
                                        ]].values.repeat(neg_sample, axis=0)))
        prev_reward_matrix = sp.csr_matrix(
            reward_scaler.transform(df[['previousReward'
                                        ]].values.repeat(neg_sample, axis=0)))
        # encode negtive feature
        neg_item_matrix = item_encoder.transform(neg_df[['competitionId']])
        neg_reward_matrix = sp.csr_matrix(neg_df[['rewardQuantity']])

        pos_matrix = sp.hstack([
            user_matrix, prev_matrix, item_matrix, cat_matrix, cont_matrix,
            prev_reward_matrix, reward_matrix
        ])
        neg_matrix = sp.hstack([
            user_matrix, prev_matrix, neg_item_matrix, cat_matrix, cont_matrix,
            prev_reward_matrix, neg_reward_matrix
        ])

        # encode period feature
        per_array = period_encoder.transform(df[['period'
                                                 ]].values.repeat(neg_sample,
                                                                  axis=0))

        user_tensor = T.tensor(user_matrix.indices,
                               dtype=T.long,
                               device=device)
        pos_tensor = self._build_feat_tensor(pos_matrix, device=device)
        neg_tensor = self._build_feat_tensor(neg_matrix, device=device)
        per_tensor = T.tensor(per_array, dtype=T.double, device=device)
        per_tensor = per_tensor.squeeze()

        return user_tensor, pos_tensor, neg_tensor, per_tensor


if __name__ == "__main__":
    item_path = Path("./inputs/kaggle/item.csv")

    databunch = SeqKaggle(item_path)
    train_ld = databunch.get_dataloader(ds_type='train', collate_fn='seq')
    train_it = iter(train_ld)

    user_tensor, pos_tensor, neg_tensor, per_tensor = next(train_it)
    print(user_tensor.size())
    print(per_tensor.size())
