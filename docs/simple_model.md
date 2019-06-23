# Simple Model

## Data Preprocessing(dataset/movielen.py)

- split train and test data:
  - for recommendation system, we need to use the **latest data** for validation and test. Therefore, I use df.duplicate() function to find most recent user, movie pair for test dataset after sorting the dataframe by timestamp
- preprocess
  - build category dict:
    - the key is the columns name and the value is the categorical column vocabulary which means that how many type of the value in current column
  - transform dataframe to tensorflow dataset
    - pop up label column
    - use df.to_dict() for converting dataframe to python dictionary object
    - build up tf.data.Dataset
  - build one-hot encoding label
    - Since it's multi-class classfy problem and keras require the label tensor is one-hot represent which means the shape of label is (num_sample, num_label_type). Note: keras.utils.to_categorical fail to execute in tf.dataset.map function

## Simple Model

- Feature Layer: we use tf API feature column to build embedding layer for userId and movieId columns
- Dense Layer: simple linear layer conbine with relu activation function. finaly the model will generate the output respect to the shape (num_batch, num_classes)

## Shortcoming

- The accuarcy is really low
- Don't use timestamp attribution
- If new item and new user coming into the system the model cannot handle the problem
- Don't use the attrubiton, related to movie and the use
