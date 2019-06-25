from pathlib import Path
from unittest import TestCase

import pandas as pd

from datasets.movielen import Movielen10K
from models import SimpleModel, ItemPopularity

MOVIE10K_PATH = Path('inputs/ml-100k/u.data')


class TestModel(TestCase):
    def test_simple_model(self):
        dataset = Movielen10K(MOVIE10K_PATH)
        vocab_dict = dataset.get_vocab_dict()
        num_classes = dataset.num_classes
        emb_dict = {'userId': 16, 'movieId': 16}
        model = SimpleModel(vocab_dict, emb_dict, num_classes)

    def test_item_pop(self):
        data = pd.DataFrame(data={'item': [1, 2, 1, 2, 3, 3, 1]})
        model = ItemPopularity(item_name='item')
        model.fit(data)
        print(model.items_pop)