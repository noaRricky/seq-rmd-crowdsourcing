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
        self.items_pop = df[item_name].value_counts(
            ascending=False).index[:self.topk]

    def evalute(self, test_df: pd.DataFrame) -> None:
        test_series = test_df[self.item_name]
        self.percision = self._precision(test_series, self.items_pop)
        self.mean_recirocal_rank = self._mean_recirocal_rank(
            test_series, self.items_pop)

    def _precision(self, test_data: pd.Series,
                   rmd_data: pd.Int64Index) -> float:
        count_series = test_data.value_counts()
        right_num = 0
        for each_item in rmd_data:
            if each_item in count_series.index:
                right_num += count_series[each_item]
        record_num = test_data.size
        return right_num / record_num

    def _mean_recirocal_rank(self, test_data: pd.Series,
                             rmd_data: pd.Int64Index) -> float:
        record_num = test_data.size
        count_series = test_data.value_counts()
        right_array = np.zeros(shape=(self.topk))
        for index, each_item in enumerate(rmd_data):
            if each_item in count_series.index:
                right_array[index] = count_series[each_item]
        mmr_array = right_array / np.arange(start=1, stop=self.topk + 1)
        mmr_rate = np.sum(mmr_array)
        return mmr_rate / record_num