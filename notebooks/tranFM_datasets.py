#%%
import pandas as pd
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from typing import Dict, List
import copy
import os

#%%
user_min = 5
item_min = 5

df = pd.read_csv("inputs/ml-100k/u.data",
                 header=None,
                 sep="\t",
                 names=["userId", "itemId", "rating", "time"])
df.head()

#%%
print('First pass')
print('num_users = {}'.format(df["userId"].unique().size))
print('num_items = {}'.format(df["itemId"].unique().size))
print('df_shape  = {}'.format(df.shape))

#%%
user_counts = df["userId"].value_counts()
user_counts.head()

#%%
item_counts = df["itemId"].value_counts()
item_counts.head()

#%%
df = df[df["userId"] >= user_min]
df = df[df["itemId"] >= item_min]

#%%
print('Second pass')
print('num_users = {}'.format(df["userId"].unique().size))
print('num_items = {}'.format(df["itemId"].unique().size))
print('df_shape  = {}'.format(df.shape))

#%%
# Normalize teporal values
print("Normalizing temporal values...")
mean_time = df["time"].mean()
std_time = df["time"].std()
df["time"] = (df["time"] - mean_time) / std_time

#%%
ONE_YEAR = (60 * 60 * 24 * 365) / mean_time
ONE_DAY = (60 * 60 * 24) / mean_time

#%%
print("Constructing datasets ...")
training_set: Dict[int, List] = defaultdict(list)
# Start counting users and items at 1 to facilitate sparse matrix computation
num_users = 1
num_items = 1
item_to_idx: Dict[int, int] = {}
user_to_idx: Dict[int, int] = {}
idx_to_item: Dict[int, int] = {}
idx_to_user: Dict[int, int] = {}

#%%
# Map ite fro user to set of items for easy look up
item_set_per_user: Dict[int, set] = {}

#%%
for row in df.itertuples():
    # New item
    if row.item_id not in item_to_idx:
        item_to_idx[row.item_id] = num_items
        idx_to_item[num_items] = row.item_id
        num_items += 1

    # New user
    if row.user_id not in user_to_idx:
        user_to_idx[row.user_id] = num_users
        idx_to_user[num_users] = row.user_id
        num_users += 1

    # Converts all ratings to positive implicit feedback
    training_set[user_to_idx[row.user_id]].append(
        (item_to_idx[row.item_id], row.time))

for user in training_set:
    training_set[user].sort(key=lambda x: x[1])

#%%
