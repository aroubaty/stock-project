# %matplotlib tk

#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pandas.plotting import register_matplotlib_converters
import os
register_matplotlib_converters()

# Disable tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

OUTPUT_FOLDER = './stock-project-output/'

SCALER = MinMaxScaler(feature_range=(0, 1))

# trainSize = -1 => take all data
def Shape_Data(rawDataFile, trainSize = -1):
    datasetDataFrame = pd.DataFrame(index=range(0,len(rawDataFile)),columns=['Date', 'Close'])

    for i in range(0,len(rawDataFile)):
        datasetDataFrame['Date'][i] = rawDataFile['Date'][i]
        datasetDataFrame['Close'][i] = rawDataFile['Close'][i] 

    #setting index
    datasetDataFrame.index = datasetDataFrame.Date
    datasetDataFrame.drop('Date', axis=1, inplace=True)

    #creating train and test sets
    dataset = datasetDataFrame.values

    train = dataset[0:987,:]
    # valid = dataset[987:,:]

    # put value between 0 and 1
    #converting dataset into x_train and y_train
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = SCALER.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0]) # group the first colum of the array by 60
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train) # change the type of array

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # transform in cube of 927,60,1

    return [x_train, y_train, datasetDataFrame]

def Predict(model, datasetDataFrame, predictionLen):
    inputs = datasetDataFrame[len(datasetDataFrame) - predictionLen - 60:].values
    inputs = inputs.reshape(-1,1)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    inputs  = SCALER.transform(inputs) # scale between 0 / 1 

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test) # natrix of 248,60

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # transform in cube of 248,60,1
    prediction = model.predict(X_test)
    prediction = SCALER.inverse_transform(prediction)
    return prediction

def StockProject(stockDataFile):
    print('----------- Load data file : ', stockDataFile)
    rawDataFile = pd.read_csv('UBSG.SW.csv')
    rawDataFile.head()
    rawDataFile['Date'] = pd.to_datetime(rawDataFile.Date,format='%Y-%m-%d')
    rawDataFile.index = rawDataFile['Date']

    # save raw chart
    plt.figure(figsize=(16,8))
    plt.plot(rawDataFile['Close'], label='Close Price history')
    plt.savefig(OUTPUT_FOLDER + 'rawDataFile.png')

    rawDataFile = rawDataFile.sort_index(ascending=True, axis=0)

    [x_train, y_train, datasetDataFrame] = Shape_Data(rawDataFile)

    # train model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2) # train the model here

    prediction = Predict(model, datasetDataFrame, 269)

    print('datasetDataFrame[987:] => ', len(datasetDataFrame[987:]))
    print('prediction => ', len(prediction))

    valid = datasetDataFrame[986:]
    valid['Predictions'] = prediction
    ## plt.plot(train['Close'])
    # plt.plot(valid[['Close','Predictions']])
    plt.plot(valid[['Predictions']])
    plt.savefig(OUTPUT_FOLDER + 'prediction.png')

    pass

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

StockProject('UBSG.SW.csv')