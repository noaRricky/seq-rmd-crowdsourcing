from pathlib import Path
from typing import List, Dict, Optional
from ast import literal_eval

import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import torch as T
from torch.utils.data import DataLoader

from utils import build_logger
from datasets.base import DFDataset, DataBunch

logger = build_logger()


class TorchTopcoder(DataBunch):
    def __init__(self,
                 regs_path: Path,
                 chag_path: Optional[Path] = None,
                 regs_min: int = 4,
                 chag_min: int = 4) -> None:
        self._read_regs_csv(regs_path, regs_min, chag_min)
        if chag_path:
            self._read_chag_csv(chag_path)

        self.batch_size = 32
        self.shuffle = False
        self.num_workers = 0
        self.device = T.device('cpu')
        self.feat_dim = self._regs_size + 2 * self._chag_size
        if chag_path:
            self.feat_dim += 2 * self._seq_size

        self._regs_path = regs_path
        self._chag_path = chag_path

    def _read_regs_csv(self, regs_path: Path, regs_min: int,
                       chag_min: int) -> None:
        # Read dataframe
        regs_df = pd.read_csv(regs_path)
        regs_df['date'] = pd.to_datetime(regs_df['date'],
                                         infer_datetime_format=True)

        # Print information
        logger.info(f"Read dataset in {regs_path}")
        logger.info(f"Original regs shape: {regs_df.shape}")

        # get counting information
        regs_counts = regs_df['registant'].value_counts()
        chag_counts = regs_df['challengeId'].value_counts()
        logger.info(f"Original registants size: {regs_counts.size}")
        logger.info(f"Original challenges size: {chag_counts.size}")

        # remove sparse item in counts
        regs_counts = regs_counts[regs_counts >= regs_min]

        # Remove sparse item
        regs_df = regs_df[regs_df['registant'].isin(regs_counts.index)]
        logger.info(f"Filter dataframe shape: {regs_df.shape}")

        # Add previous columns
        regs_df = regs_df.sort_values(by=['registant', 'date'])
        regs_df['previousId'] = regs_df['challengeId']

        # Shift previous column
        regs_df['previousId'] = regs_df['previousId'].shift(
            periods=1).fillna(0).astype('int64')

        # Set first item non for each user
        regs_df = regs_df.sort_values(by=['registant', 'date'])
        first_mask = ~regs_df.duplicated(subset=['registant'], keep='first')
        regs_df['previousId'][first_mask] = -1

        # Add encoder
        chag_encoder = OneHotEncoder(categories='auto',
                                     handle_unknown='ignore')
        regs_encoder = OneHotEncoder(categories='auto', handle_unknown='error')
        chag_encoder.fit(regs_df[['challengeId']])
        regs_encoder.fit(regs_df[['registant']])

        # Split dataset to train, valid, test
        regs_df = regs_df.sort_values(by=['date'])

        last_mask = regs_df.duplicated(subset=['registant'], keep='last')
        remain_df = regs_df[last_mask]
        test_df = regs_df[~last_mask]

        last_mask = remain_df.duplicated(subset=['registant'], keep='last')
        train_df = remain_df[last_mask]
        valid_df = remain_df[~last_mask]

        cat_name = ['registant', 'challengeId', 'previousId']

        self._df_dict: Dict[str, pd.DataFrame] = {
            'train': train_df[cat_name],
            'valid': valid_df[cat_name],
            'test': test_df[cat_name]
        }
        self._regs_encoder = regs_encoder
        self._chag_encoder = chag_encoder
        self.user_size = regs_encoder.categories_[0].size
        self._regs_size = regs_encoder.categories_[0].size
        self._chag_size = chag_encoder.categories_[0].size

    def _read_chag_csv(self, chag_path: Path) -> None:
        # Read challenge dataframe
        chag_df = pd.read_csv(chag_path,
                              converters={
                                  'technologies': literal_eval,
                                  'platforms': literal_eval
                              })
        chag_df['date'] = pd.to_datetime(chag_df['date'],
                                         infer_datetime_format=True)
        chag_df = chag_df.set_index('challengeId')
        chag_df = chag_df[['technologies', 'platforms']]

        # build encoder
        tech_binarizer = MultiLabelBinarizer(sparse_output=True)
        plat_binarizer = MultiLabelBinarizer(sparse_output=True)
        tech_binarizer.fit(chag_df['technologies'].tolist())
        plat_binarizer.fit(chag_df['platforms'].tolist())

        # Append non-challenge info
        chag_df.loc[-1] = pd.Series({'technologies': [], 'platforms': []})

        self._chag_df = chag_df
        self._tech_binarizer = tech_binarizer
        self._plat_binarizer = plat_binarizer
        self._seq_size = tech_binarizer.classes_.size + plat_binarizer.classes_.size

    def _data_collate(self, batch: List[pd.Series]):
        device = self.device
        regs_encoder = self._regs_encoder
        chag_encoder = self._chag_encoder

        df = pd.DataFrame(batch)
        row_num = df.shape[0]

        # Generate negtive column
        df['negtiveId'] = np.random.choice(a=chag_encoder.categories_[0],
                                           size=row_num)

        # Encode feature
        regs_vector: sp.csr_matrix = regs_encoder.transform(df[['registant']])
        chag_vector = chag_encoder.transform(df[['challengeId']])
        prev_vector = chag_encoder.transform(df[['previousId']])
        negi_vector = chag_encoder.transform(df[['negtiveId']])

        pos_feat_matrix = sp.hstack([regs_vector, prev_vector, chag_vector])
        neg_feat_matrix = sp.hstack([regs_vector, prev_vector, negi_vector])

        if self._chag_path:
            chag_df = self._chag_df
            tech_binarizer = self._tech_binarizer
            plat_binarizer = self._plat_binarizer

            batch_chag_df = chag_df.loc[df['challengeId']]
            chag_tech_matrix = tech_binarizer.transform(
                batch_chag_df['technologies'])
            chag_plat_matrix = plat_binarizer.transform(
                batch_chag_df['platforms'])
            batch_prev_df = chag_df.loc[df['previousId']]
            prev_tech_matrix = tech_binarizer.transform(
                batch_prev_df['technologies'])
            prev_plat_matrix = plat_binarizer.transform(
                batch_prev_df['platforms'])
            batch_negi_df = chag_df.loc[df['negtiveId']]
            negi_tech_matrix = tech_binarizer.transform(
                batch_negi_df['technologies'])
            negi_plat_matrix = plat_binarizer.transform(
                batch_negi_df['platforms'])

            pos_feat_matrix = sp.hstack([
                pos_feat_matrix, prev_tech_matrix, prev_plat_matrix,
                chag_tech_matrix, chag_plat_matrix
            ])
            neg_feat_matrix = sp.hstack([
                neg_feat_matrix, prev_tech_matrix, prev_plat_matrix,
                negi_tech_matrix, negi_plat_matrix
            ])

        user_tensor = T.tensor(regs_vector.indices, dtype=T.long)
        pos_tensor = self._build_feat_tensor(pos_feat_matrix)
        neg_tensor = self._build_feat_tensor(neg_feat_matrix)

        user_tensor = user_tensor.to(device)
        pos_tensor = pos_tensor.to(device)
        neg_tensor = neg_tensor.to(device)

        return user_tensor, pos_tensor, neg_tensor

    def _build_feat_tensor(self,
                           feat_matrix: sp.coo_matrix,
                           device: Optional[T.device] = T.device('cpu')):
        feat_index_list = feat_matrix.nonzero()
        feat_index_array: np.ndarray = np.vstack(feat_index_list)

        feat_index = T.tensor(feat_index_array.tolist(),
                              dtype=T.long,
                              device=device)
        feat_value = T.tensor(feat_matrix.data, dtype=T.double, device=device)
        feat_tensor = T.sparse_coo_tensor(feat_index,
                                          feat_value,
                                          size=feat_matrix.shape,
                                          device=device)

        return feat_tensor


if __name__ == '__main__':
    REG_PATH = Path("../inputs/topcoder/regs.csv")
    CHA_PATH = Path("../inputs/topcoder/challenge.csv")
    databunch = TorchTopcoder(regs_path=REG_PATH, chag_path=CHA_PATH)
