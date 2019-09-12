import unittest
from pathlib import Path

import torch as T

from datasets import TorchTopcoder

class MyTestCase(unittest.TestCase):
    def test_something(self):
        DEVICE = T.device('cpu')
        BATCH = 3
        SHUFFLE = True
        WORKER_NUM = 0
        REGS_PATH = Path("../inputs/topcoder/regs.csv")
        CHAG_PATH = Path("../inputs/topcoder/challenge.csv")

        databunch = TorchTopcoder(regs_path=REGS_PATH, chag_path=CHAG_PATH)
        train_dl = databunch.get_dataloader(dataset_type='train')
        train_it = iter(train_dl)
        user_tenser, pos_tensor, neg_tensor = next(train_it)
        print(f"pos tensor shape: {pos_tensor.shape}")
        print(f"neg tensor shape: {neg_tensor.shape}")
        print(f"feature dimension: {databunch.feat_dim}")


if __name__ == '__main__':
    unittest.main()
