from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column, keras


class Movielen10K(object):
    def __init__(self, data_path: Path) -> None:
        self.num_classes = 5
        self._COLUMNS = ['userId', 'movieId', 'rating', 'timestamp']
        self._CAT_NAMES = ['userId', 'movieId']
        # Read 10k data
        raw_df = pd.read_csv(data_path,
                             sep='\t',
                             names=self._COLUMNS,
                             engine='python')
        dict_df = self._split_train_test(raw_df)
        train_size = dict_df['train'].shape[0]
        test_size = dict_df['test'].shape[0]
        train_ds = self._df_to_dataset(dict_df['train'])
        test_ds = self._df_to_dataset(dict_df['test'])
        self._dict_size: Dict[str, int] = {
            'train': train_size,
            'test': test_size
        }
        self._dict_ds: Dict[str, tf.data.Dataset] = {
            'train': train_ds,
            'test': test_ds
        }
        self._dict_vocab = self._build_vocab_dict(dict_df['train'],
                                                  self._CAT_NAMES)

    def get_dataset(self, ds_type: str, batch_size: int,
                    shuffle: bool = True) -> tf.data.Dataset:
        assert ds_type in self._dict_ds, "don't contain {} dataset".format(
            ds_type)
        ds = self._dict_ds[ds_type]
        ds = ds.batch(batch_size)
        if shuffle:
            ds = ds.shuffle(self._dict_size[ds_type])
        ds = ds.map(self._preprocess)
        return ds

    def get_vocab_dict(self) -> Dict[str, np.ndarray]:
        return self._dict_vocab

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

    def _preprocess(self, features: Dict[str, tf.Tensor],
                    labels: tf.Tensor) -> Any:
        labels = labels - 1
        labels = tf.one_hot(labels, self.num_classes)
        return features, labels

    def _build_vocab_dict(self, df: pd.DataFrame,
                          cat_names: List[str]) -> Dict[str, np.ndarray]:
        return {name: df[name].unique() for name in cat_names}


class Movielen10M(object):
    def __init__(self, data_path: Path) -> None:
        pass


if __name__ == "__main__":
    pass
