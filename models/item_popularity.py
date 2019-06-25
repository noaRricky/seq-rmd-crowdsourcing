import numpy as np
import pandas as pd


class ItemPopularity(object):
    def __init__(self, item_name: str, topk: int = 20) -> None:
        self.item_name = item_name
        self.items_pop: pd.DataFrame = None
        self.topk = topk
        self.percision: float
        self.mean_recirocal_rank: float

    def fit(self, df: pd.DataFrame) -> None:
        item_name = self.item_name
        self.items_pop = df[item_name].sort_values(
            ascending=False).index[:self.topk]

    def evalute(self, test_df: pd.DataFrame) -> None:
        test_series = test_df[self.item_name]
        self.percision = self._precision(test_series, self.items_pop)

    def _precision(self, test_data: pd.Series,
                   rmd_data: pd.Int64Index) -> float:
        right_num = test_data.sort_values()[rmd_data].sum()
        record_num = test_data.size
        return right_num / record_num

    def _mean_recirocal_rank(self, test_data: pd.Series,
                             rmd_data: pd.Int64Index) -> float:
        record_num = test_data.size
        right_array = test_data.value_counts()[rmd_data].to_numpy()
        mmr_array = right_array / np.arange(start=1, stop=self.topk + 1)
        mmr_rate = np.sum(mmr_array)
        return mmr_rate / record_num