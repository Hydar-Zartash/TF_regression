from setup import *
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    pick = input("Stock pick: ")
    stock = StockSetup(pick, 3)
except Exception as e:
    print(e)

#split into train and test sets. 
#Because I went for a more isolated approach where each daily state is linked to a boolean value of 
#if the stock grew by X amount, we do not necessairily need to use a time-series forecasting method

train_set = stock.data.sample(frac=0.75, random_state=0)
test_set = stock.data.drop(train_set.index)

#print(train_set.describe().transpose())

train_features = train_set.copy()
test_features = test_set.copy()

train_label = train_features.pop("Growth X%")
test_label = test_features.pop("Growth X%")

nrm = layers.experimental.preprocessing.Normalization(axis=-1) # using an alias because layers.Normalization doesnt work for some reason