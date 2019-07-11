# Note for transFM Code

## Dataset result

- Contents
  - user_to_idx: Dict[user_id, index from 0 to user_size]
  - idx_to_user: Dict[index from 0 to user_size, user_id]
  - item_to_idx
  - idx_to_item
  - traning_set: Dict[user_index, (item_index, item_time)]

## New Preprocess step

- add prev item
- split train, valid, test
- get categories
- add first nan item
