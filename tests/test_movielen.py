from pathlib import Path
from unittest import TestCase

from datasets.movielen import Movielen10K

MOVIE10K_PATH = Path('inputs/ml-100k/u.data')


class TestMovelen(TestCase):
    def test_movielen10k(self):
        movelen10k_dataset = Movielen10K(MOVIE10K_PATH)
        train_ds = movelen10k_dataset.train_ds
        test_ds = movelen10k_dataset.test_ds

        train_ds = train_ds.batch(32)
        for feature_batch, label_batch in train_ds.take(1):
            print('batch of userId: {}'.format(feature_batch['userId']))
            print('batch of label: {}'.format(label_batch))

    def test_movielen10k_dataset(self):
        movielen10k_dataset = Movielen10K(MOVIE10K_PATH)
        train_ds = movielen10k_dataset.get_dataset('train',
                                                   batch_size=32,
                                                   shuffle=True)
        test_ds = movielen10k_dataset.get_dataset('test',
                                                  batch_size=32,
                                                  shuffle=False)

        for feature_batch, label_batch in train_ds.take(1):
            print('batch of feature: {}'.format(feature_batch))
            print('batch of userId: {}'.format(feature_batch['userId']))
            print('batch of label: {}'.format(label_batch))

    def test_movielen10k_vocab(self):
        dataset = Movielen10K(MOVIE10K_PATH)
        vocab_dict = dataset.get_vocab_dict()
        print(f"user array shape: {vocab_dict['userId'].shape}")
        print(f"movie array shape: {vocab_dict['movieId'].shape}")
