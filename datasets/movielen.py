from pathlib import Path
from typing import Dict, List

import pandas as pd
import tensorflow as tf
from tensorflow import feature_column


class Movielen10K(object):
    def __init__(self, data_path: Path) -> None:
        self._COLUMNS = ['userId', 'movieId', 'timestamp', 'rating']
        # Read 10k data
        raw_df = pd.read_csv(data_path,
                             sep='\t',
                             names=self._COLUMNS,
                             engine='python')
        dict_df = self._split_train_test(raw_df)
        self.train_ds = self._df_to_dataset(dict_df['train'])
        self.test_ds = self._df_to_dataset(dict_df['test'])

    def _split_train_test(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df.sort_values(by=['timestamp'], inplace=True)
        last_mask = df.duplicated(subset=['userId'], keep='last')

        train_df: pd.DataFrame = df[last_mask]
        test_df: pd.DataFrame = df[~last_mask]

        train_df.sort_values(by=['userId', 'timestamp'], inplace=True)
        test_df.sort_values(by=['userId', 'timestamp'], inplace=True)
        return {'train': train_df, 'test': test_df}

    def _df_to_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        df = df.copy()
        # Delete timestamp while I currenly don't use it
        df.pop('timestamp')
        labels = df.pop('rating')
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        return ds


class Movielen10M(object):
    def __init__(self, data_path: Path) -> None:
        pass


if __name__ == "__main__":
    pass
