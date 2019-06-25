from typing import Dict
from pathlib import Path
from unittest import TestCase

import pandas as pd

from datasets.movielen import Movielen10K
from models import SimpleModel, ItemPopularity

MOVIE10K_PATH = Path('inputs/ml-100k/u.data')


def split_train_test(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df.sort_values(by=['timestamp'], inplace=True)
    last_mask = df.duplicated(subset=['userId'], keep='last')

    train_df: pd.DataFrame = df[last_mask]
    test_df: pd.DataFrame = df[~last_mask]

    train_df.sort_values(by=['userId', 'timestamp'], inplace=True)
    test_df.sort_values(by=['userId', 'timestamp'], inplace=True)
    return {'train': train_df, 'test': test_df}


class TestModel(TestCase):
    def test_simple_model(self):
        dataset = Movielen10K(MOVIE10K_PATH)
        vocab_dict = dataset.get_vocab_dict()
        num_classes = dataset.num_classes
        emb_dict = {'userId': 16, 'movieId': 16}
        model = SimpleModel(vocab_dict, emb_dict, num_classes)

    def test_item_pop(self):
        df = pd.read_csv('inputs/ml-100k/u.data',
                         sep='\t',
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         engine='python')
        model = ItemPopularity(item_name='movieId', topk=20)
        model.fit(df)
        model.evalute(df)