import os
from setup import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




def build_compile(nrm) -> keras.Sequential:
    """Builds a sequential model of two relu dense layers and a sigmoid binary layer as output

    Args:
        nrm (Normalization): fitted training dataset

    Returns:
        keras.Sequential: sequential model of two relu dense layers and a sigmoid binary layer
    """
    mdl = keras.Sequential([
        nrm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation = 'sigmoid') #we only have a single output, so our final layer hasa single node
        ])

# learning rate determined through trial and error until overfitting was not detected from loss
    mdl.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001)) 


    return mdl

def plt_loss(history) -> None:  
    '''
    utility function that plots our models error over time (epoch)
    helps look for overfitting 
    '''
    plt.plot(history.history['loss'], label ='loss')
    plt.plot(history.history['val_loss'], label = 'validation_loss')

    plt.ylim([0,0.5])

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Error')

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ =="__main__":
    try:
        pick = input("Stock pick: ")
        target = int(input("Target growth percentage for a month: "))
        stock = StockSetup(pick, target)
    except Exception as e:
        print(e)

    #split into train and test sets. 
    #Because I went for a more isolated approach where each daily state is linked to a boolean value of 
    #if the stock grew by X amount, we do not necessairily need to use a time-series forecasting method

    train_set = stock.data.sample(frac=0.75, random_state=0)
    test_set = stock.data.drop(train_set.index)




    train_features = train_set.copy()
    test_features = test_set.copy()

    train_label = train_features.pop("Growth X%")
    test_label = test_features.pop("Growth X%")

    norm = layers.experimental.preprocessing.Normalization(axis=-1) # using an alias because layers.Normalization doesnt work for some reason
    norm.adapt(np.array(train_features)) #fit state of preprocessing layer to data

    # start Regression analysis with DNN

    dnn_mdl = build_compile(norm)



    history = dnn_mdl.fit(
        x=train_features, 
        y = train_label,
        validation_split=0.2,
        verbose=0,
        epochs=100
    )



    test_result = {}

    test_result['model'] = dnn_mdl.evaluate(test_features[:-30], test_label[:-30], verbose = 0)

    test_prdct = dnn_mdl.predict(test_features).flatten()


    error = test_prdct - test_label

    print("\n\n\n\n RESULTS \n\n")
    print(f"{pick.upper()} confidence to grow {target} percent in the next month : {round(test_prdct[-5:].mean()*100, 2)}%")
    print(f"estimator error :{round(abs(error.mean()) * 100,2)}%")


