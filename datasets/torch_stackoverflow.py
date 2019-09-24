from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MultiLabelBinarizer
import torch as T
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from datasets.base import DFDataset, DataBunch


class TorchStackOverflow(DataBunch):
    def __init__(self, data_path: Path, min_user: Optional[int] = 4):
        item_df: pd.DataFrame = pd.read_csv(data_path)

        # Change data types
        item_df['time'] = pd.to_datetime(item_df['time'],
                                         infer_datetime_format=True)

        # Filter data
        user_counts: pd.Series = item_df['userId'].value_counts()
        print(f"Original user size: {user_counts.size}")
        user_counts = user_counts[user_counts >= min_user]
        print(f"Filtered user size: {user_counts.size}")

        print(f"Original item dataframe shape: {item_df.shape}")
        item_df = item_df[item_df['userId'].isin(user_counts.index)]
        print(f"Filtered item dataframe shape: {item_df.shape}")

        # Generate question dataframe
        ques_cols = ['questionId', 'tags', 'answers', 'views', 'questionVote']
        ques_df = item_df[ques_cols]
        ques_df = ques_df.drop_duplicates(subset=['questionId'])
        # Add default row
        ques_df.loc[-1] = [0, '', 0, 0, 0]
        # Change tags column
        ques_df['tags'] = ques_df['tags'].str.split(' ')

        # Drop question information columns in item dataframe
        item_df = item_df[['questionId', 'userId', 'time', 'answerVote']]

        # Generate previous columns
        prev_cols = ['prevQuesId', 'answerVote']
        item_df['prevQuesId'] = item_df['questionId']
        item_df = item_df.sort_values(by=['userId', 'time'])
        item_df[prev_cols] = item_df[prev_cols].shift(periods=1, fill_value=0)
        # Fill first random information for each user
        first_mask = item_df.duplicated(subset=['userId'], keep='first')
        item_df[prev_cols] = item_df[prev_cols].where(first_mask, 0)

        # Add encoder
        cont_names = ['answers', 'views', 'questionVote']
        question_encoder = OneHotEncoder(categories='auto',
                                         handle_unknown='ignore')
        user_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        ans_vote_scaler = MinMaxScaler(feature_range=(0, 1))
        cont_scaler = MinMaxScaler(feature_range=(0, 1))
        tags_binzer = MultiLabelBinarizer(sparse_output=True)
        # Fit all encoder
        question_encoder.fit(item_df[['questionId']])
        user_encoder.fit(item_df[['userId']])
        ans_vote_scaler.fit(item_df[['answerVote']])
        cont_scaler.fit(ques_df[cont_names])
        tags_binzer.fit(ques_df['tags'])

        ## Split train, valie and test dataframe
        # Split test dataframe
        item_df = item_df.sort_values(by=['userId', 'time'])
        last_masks = item_df.duplicated(subset=['userId'], keep='last')
        remain_df = item_df[last_masks]
        test_df = item_df[~last_masks]
        # Split train, valid dataframe
        last_masks = remain_df.duplicated(subset=['userId'], keep='last')
        train_df = remain_df[last_masks]
        valid_df = remain_df[~last_masks]

        # Generate feature dimension
        question_dim = question_encoder.categories_[0].size
        user_dim = user_encoder.categories_[0].size
        ans_vote_dim = 1
        cont_dim = len(cont_names)
        tags_dim = tags_binzer.classes_.size

        # Setup protected attributions
        self._df_dict = {'train': train_df, 'valid': valid_df, 'test': test_df}
        self._ques_df = ques_df
        self._question_encoder = question_encoder
        self._user_encoder = user_encoder
        self._ans_vote_scaler = ans_vote_scaler
        self._cont_scaler = cont_scaler
        self._tags_binzer = tags_binzer
        self._cont_names = cont_names

        # Set up public attributions
        self.feat_dim = user_dim + 2 * question_dim + ans_vote_dim + 2 * cont_dim + 2 * tags_dim
        self.batch_size = 32
        self.shuffle = False
        self.num_workers = 0
        self.device = T.device('cpu')

    def _data_collate(self, batch: List[pd.Series]):
        device = self.device
        cont_names = self._cont_names
        question_encoder = self._question_encoder
        user_encoder = self._user_encoder
        ans_vote_scaler = self._ans_vote_scaler
        cont_scaler = self._cont_scaler
        tags_binzer = self._tags_binzer
        ques_df = self._ques_df

        # Generate feature dataframe
        df = pd.DataFrame(batch)
        pos_df: pd.DataFrame = pd.merge(left=df[['questionId']],
                                        right=ques_df,
                                        on='questionId',
                                        how='inner')
        prev_df: pd.DataFrame = pd.merge(left=df[['prevQuesId']],
                                         right=ques_df,
                                         left_on='prevQuesId',
                                         right_on='questionId',
                                         how='inner')
        neg_df = ques_df.sample(n=len(batch))

        # encode features
        user_mat: sp.csr_matrix = user_encoder.transform(df[['userId']])
        prev_ques_mat = question_encoder.transform(df[['prevQuesId']])
        pos_ques_mat = question_encoder.transform(df[['questionId']])
        neg_ques_mat = question_encoder.transform(neg_df[['questionId']])
        ans_vote_mat = ans_vote_scaler.transform(df[['answerVote']])
        prev_tags_mat = tags_binzer.transform(prev_df['tags'])
        pos_tags_mat = tags_binzer.transform(pos_df['tags'])
        neg_tags_mat = tags_binzer.transform(neg_df['tags'])
        prev_cont_mat = sp.csr_matrix(
            cont_scaler.transform(prev_df[cont_names]))
        pos_cont_mat = sp.csr_matrix(cont_scaler.transform(pos_df[cont_names]))
        neg_cont_mat = sp.csr_matrix(cont_scaler.transform(neg_df[cont_names]))

        # stack different features
        pos_mat = sp.hstack([
            user_mat, prev_ques_mat, pos_ques_mat, ans_vote_mat, prev_tags_mat,
            prev_cont_mat, pos_tags_mat, pos_cont_mat
        ])
        neg_mat = sp.hstack([
            user_mat, prev_ques_mat, neg_ques_mat, ans_vote_mat, prev_tags_mat,
            prev_cont_mat, neg_tags_mat, neg_cont_mat
        ])

        user_tensor = T.tensor(user_mat.indices, dtype=T.long, device=device)
        pos_tensor = self._build_feat_tensor(pos_mat, device=device)
        neg_tensor = self._build_feat_tensor(neg_mat, device=device)

        return user_tensor, pos_tensor, neg_tensor


if __name__ == "__main__":
    data_path = Path("./inputs/stackoverflow/item.csv")

    databunch = TorchStackOverflow(data_path, min_user=4)
    train_ld = databunch.get_dataloader(ds_type='train')
    train_it = iter(train_ld)
    user_tensor, pos_tensor, neg_tensor = next(train_it)
    print(pos_tensor.shape)
    print(databunch.feat_dim)
