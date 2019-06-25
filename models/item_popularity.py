import pandas as pd


class ItemPopularity(object):
    def __init__(self, item_name: str, topk: int = 10) -> None:
        self.item_name = item_name
        self.items_pop: pd.DataFrame = None
        self.topk = topk

    def fit(self, data: pd.DataFrame) -> None:
        item_name = self.item_name
        items_pop = data[[item_name]].groupby(by=[item_name]).size()
        self.items_pop = items_pop.sort_values(ascending=False)[:self.topk]

    def evalute(self, test_data) -> None:
        pass