from typing import List, Dict

import tensorflow as tf
from tensorflow import keras, feature_column

from datasets.movielen import Movielen10K


class SimpleModel(keras.Model):
    def __init__(self, vocab_dict: Dict[str, int], emb_dict: Dict[str, int],
                 num_classes: int) -> None:
        super(SimpleModel, self).__init__(name='SimpleModel')
        self._feature_layer = self._build_feature_layer(vocab_dict, emb_dict)
        self._dense1 = keras.layers.Dense(128, activation='relu')
        self._dense2 = keras.layers.Dense(128, activation='relu')
        self._final_dense = keras.layers.Dense(num_classes,
                                               activation='softmax')

    def call(self, inputs):
        x = self._feature_layer(inputs)
        x = self._dense1(x)
        x = self._dense2(x)
        return self._final_dense(x)

    def _build_feature_layer(self, vocab_dict: Dict[str, int],
                             emb_dict: Dict[str, int]
                             ) -> keras.layers.DenseFeatures:
        """Build feature layer for sturcture data, currently only support embedding features

        Arguments:
            vacab_list {Dict[str, int]} -- vocabulary list for each category column
            emb_dict {Dict[str, int]} -- embedding dictionary key is column name , value is dimension

        Returns:
            keras.layers.DenseFeatures -- first layer for network
        """
        emb_columns: List = []
        for name, dim in emb_dict.items():
            cat_col = feature_column.categorical_column_with_vocabulary_list(
                name, vocabulary_list=vocab_dict[name])
            emb_col = feature_column.embedding_column(cat_col, dim)
            emb_columns.append(emb_col)
        feature_layer = keras.layers.DenseFeatures(emb_columns)
        return feature_layer
