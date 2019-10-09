from pathlib import Path
from typing import List, Dict, Optional
from ast import literal_eval

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, OrdinalEncoder
import torch as T
from torch.utils.data import DataLoader

from utils import build_logger
from datasets.base import DFDataset, DataBunch

# logger = build_logger()


class SeqTopcoder(DataBunch):
    def __init__(
            self,
            regs_path: Path,
            chag_path: Path,
            regs_min: Optional[int] = 4,
    ) -> None:
        # Read dataframe
        regs_df = pd.read_csv(regs_path)

        # Print information
        print(f"Read dataset in {regs_path}")
        print(f"Original regs shape: {regs_df.shape}")

        # get counting information
        regs_counts = regs_df['registant'].value_counts()
        chag_counts = regs_df['challengeId'].value_counts()
        print(f"Original registants size: {regs_counts.size}")
        print(f"Original challenges size: {chag_counts.size}")

        # remove sparse item in counts
        regs_counts = regs_counts[regs_counts >= regs_min]

        # Remove sparse item
        regs_df = regs_df[regs_df['registant'].isin(regs_counts.index)]
        print(f"Filter dataframe shape: {regs_df.shape}")

        # Add previous and period columns
        regs_df = regs_df.sort_values(by=['registant', 'date'])
        regs_df['previousId'] = regs_df['challengeId']
        regs_df['period'] = regs_df['date'].str[:7]

        # Shift previous column
        regs_df['previousId'] = regs_df['previousId'].shift(
            periods=1).fillna(0).astype('int64')

        # Set first item non for each user
        regs_df = regs_df.sort_values(by=['registant', 'date'])
        first_mask = regs_df.duplicated(subset=['registant'], keep='first')
        regs_df['previousId'] = regs_df['previousId'].where(first_mask, -1)

        # Read attr dataframe
        chag_df: pd.DataFrame = regs_df[['challengeId', 'period']]
        chag_df = chag_df.drop_duplicates(subset=['challengeId'])
        attr_df = pd.read_csv(chag_path,
                              converters={
                                  'technologies': literal_eval,
                                  'platforms': literal_eval
                              })
        chag_df = pd.merge(left=chag_df,
                           right=attr_df,
                           how='inner',
                           on=['challengeId'])
        # Add default row
        print(chag_df.columns)
        chag_df.loc[-1] = (-1, '2005-01', '2005-01-01', 0, [], [])
        chag_df = chag_df.sort_values(by=['date'])

        # Add encoder
        chag_encoder = OneHotEncoder(categories='auto',
                                     handle_unknown='ignore')
        regs_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        period_encoder = OrdinalEncoder(categories='auto')
        tech_binarizer = MultiLabelBinarizer(sparse_output=True)
        plat_binarizer = MultiLabelBinarizer(sparse_output=True)

        chag_encoder.fit(regs_df[['challengeId']])
        regs_encoder.fit(regs_df[['registant']])
        period_encoder.fit(chag_df[['period']])
        tech_binarizer.fit(chag_df['technologies'].tolist())
        plat_binarizer.fit(chag_df['platforms'].tolist())

        # Split dataset to train, valid, test
        regs_df = regs_df.sort_values(by=['date'])

        last_mask = regs_df.duplicated(subset=['registant'], keep='last')
        remain_df = regs_df[last_mask]
        test_df = regs_df[~last_mask]

        last_mask = remain_df.duplicated(subset=['registant'], keep='last')
        train_df = remain_df[last_mask]
        valid_df = remain_df[~last_mask]

        # Add default config
        self.config_db()
        self._chag_df = chag_df
        self._df_dict: Dict[str, pd.DataFrame] = {
            'train': train_df,
            'valid': valid_df,
            'test': test_df
        }

        self._regs_encoder = regs_encoder
        self._chag_encoder = chag_encoder
        self._period_encoder = period_encoder
        self._tech_binarizer = tech_binarizer
        self._plat_binarizer = plat_binarizer

        regs_size = regs_encoder.categories_[0].size
        chag_size = chag_encoder.categories_[0].size
        seq_size = tech_binarizer.classes_.size + plat_binarizer.classes_.size

        self.feat_dim = regs_size + 2 * chag_size + 2 * seq_size
        self.user_size = regs_size

    def _base_collate(self, batch: List[pd.Series]):
        device = self.device
        regs_encoder = self._regs_encoder
        chag_encoder = self._chag_encoder
        period_encoder = self._period_encoder
        tech_binarizer = self._tech_binarizer
        plat_binarizer = self._plat_binarizer
        chag_df = self._chag_df

        df = pd.DataFrame(batch)
        prev_feat_df = pd.merge(left=df[['previousId']],
                                right=chag_df,
                                how='left',
                                left_on=['previousId'],
                                right_on=['challengeId'])
        pos_feat_df = pd.merge(left=df[['challengeId']],
                               right=chag_df,
                               how='left',
                               on=['challengeId'])
        neg_feat_df = chag_df.sample(df.shape[0])

        # encoder base feature
        regs_vector = regs_encoder.transform(df[['registant']])
        chag_vector = chag_encoder.transform(df[['previousId']])
        tech_vector = tech_binarizer.transform(prev_feat_df['technologies'])
        plat_vector = plat_binarizer.transform(prev_feat_df['platforms'])
        base_vector = sp.hstack(
            [regs_vector, chag_vector, tech_vector, plat_vector])

        # encode positive feature
        chag_vector = chag_encoder.transform(pos_feat_df[['challengeId']])
        tech_vector = tech_binarizer.transform(pos_feat_df['technologies'])
        plat_vector = plat_binarizer.transform(pos_feat_df['platforms'])
        pos_matrix = sp.hstack(
            [base_vector, chag_vector, tech_vector, plat_vector])

        # encode negtive feature
        chag_vector = chag_encoder.transform(neg_feat_df[['challengeId']])
        tech_vector = tech_binarizer.transform(neg_feat_df['technologies'])
        plat_vector = plat_binarizer.transform(neg_feat_df['platforms'])
        neg_matrix = sp.hstack(
            [base_vector, chag_vector, tech_vector, plat_vector])

        # sparse matrix to tensor
        user_tensor = T.tensor(regs_vector.indices,
                               dtype=T.long,
                               device=device)
        pos_tensor = self._build_feat_tensor(pos_matrix, device)
        neg_tensor = self._build_feat_tensor(neg_matrix, device)

        return user_tensor, pos_tensor, neg_tensor, None

    def _seq_collate(self, batch: List[pd.Series]):
        device = self.device
        regs_encoder = self._regs_encoder
        chag_encoder = self._chag_encoder
        period_encoder = self._period_encoder
        tech_binarizer = self._tech_binarizer
        plat_binarizer = self._plat_binarizer
        chag_df = self._chag_df
        neg_sample = self.neg_sample

        df = pd.DataFrame(batch)
        df = df.sort_values(by=['period'])
        prev_feat_df = pd.merge(left=df[['previousId']],
                                right=chag_df,
                                how='left',
                                left_on=['previousId'],
                                right_on=['challengeId'])
        pos_feat_df = pd.merge(left=df[['challengeId']],
                               right=chag_df,
                               how='left',
                               on=['challengeId'])

        # Generate negtive feat dataframe
        per_counts = df['period'].value_counts().sort_index()
        neg_list = [
            chag_df[chag_df['period'] == per].sample(n=per_counts[per] *
                                                     neg_sample,
                                                     replace=True)
            for per in per_counts.index
        ]
        neg_feat_df = pd.concat(neg_list)

        # encoder base feature
        regs_vector = regs_encoder.transform(df[['registant'
                                                 ]].values.repeat(neg_sample,
                                                                  axis=0))
        chag_vector = chag_encoder.transform(df[['previousId'
                                                 ]].values.repeat(neg_sample,
                                                                  axis=0))
        tech_vector = tech_binarizer.transform(
            prev_feat_df['technologies'].repeat(neg_sample))
        plat_vector = plat_binarizer.transform(
            prev_feat_df['platforms'].repeat(neg_sample))
        base_vector = sp.hstack(
            [regs_vector, chag_vector, tech_vector, plat_vector])

        # encode positive feature
        chag_vector = chag_encoder.transform(
            pos_feat_df[['challengeId']].values.repeat(neg_sample, axis=0))
        tech_vector = tech_binarizer.transform(
            pos_feat_df['technologies'].repeat(neg_sample))
        plat_vector = plat_binarizer.transform(
            pos_feat_df['platforms'].repeat(neg_sample))
        pos_matrix = sp.hstack(
            [base_vector, chag_vector, tech_vector, plat_vector])

        # encode negtive feature
        chag_vector = chag_encoder.transform(neg_feat_df[['challengeId']])
        tech_vector = tech_binarizer.transform(neg_feat_df['technologies'])
        plat_vector = plat_binarizer.transform(neg_feat_df['platforms'])
        neg_matrix = sp.hstack(
            [base_vector, chag_vector, tech_vector, plat_vector])

        # encode period feature
        period_array = period_encoder.transform(df[['period']].values.repeat(
            neg_sample, axis=0))

        # sparse matrix to tensor
        user_tensor = T.tensor(regs_vector.indices,
                               dtype=T.long,
                               device=device)
        pos_tensor = self._build_feat_tensor(pos_matrix, device)
        neg_tensor = self._build_feat_tensor(neg_matrix, device)
        per_tensor = T.tensor(period_array, dtype=T.double, device=device)
        per_tensor = per_tensor.squeeze()

        return user_tensor, pos_tensor, neg_tensor, per_tensor


if __name__ == '__main__':
    REG_PATH = Path("./inputs/topcoder/regs.csv")
    CHA_PATH = Path("./inputs/topcoder/challenge.csv")
    databunch = SeqTopcoder(regs_path=REG_PATH, chag_path=CHA_PATH)
    dl = databunch.get_dataloader(ds_type='train', collate_fn='seq')
    data_iter = iter(dl)
    user_tensor, pos_tensor, neg_tensor, per_tensor = next(data_iter)
    print(f"User size: {user_tensor.shape}")
    print(f"Pos size: {pos_tensor.shape}")
    print(f"Neg size: {neg_tensor.shape}")
    print(f"Period size: {per_tensor.shape}")
