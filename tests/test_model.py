from pathlib import Path
from unittest import TestCase

from datasets.movielen import Movielen10K
from models.simple_model import SimpleModel

MOVIE10K_PATH = Path('inputs/ml-100k/u.data')


class TestModel(TestCase):
    def test_simple_model(self):
        dataset = Movielen10K(MOVIE10K_PATH)
        vocab_dict = dataset.get_vocab_dict()
        num_classes = dataset.num_classes
        emb_dict = {'userId': 16, 'movieId': 16}
        model = SimpleModel(vocab_dict, emb_dict, num_classes)
