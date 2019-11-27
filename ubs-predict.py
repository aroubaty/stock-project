%matplotlib tk


# https://finance.yahoo.com/quote/UBSG.SW/history?period1=1416956400&period2=1574722800&interval=1d&filter=history&frequency=1d
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pandas.plotting import register_matplotlib_converters

#LOAD
#read the file
df = pd.read_csv('UBSG.SW.csv')

#print the head
df.head()
register_matplotlib_converters()
#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')

# MISE EN PLACE DES DATA
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i] 

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

# put value between 0 and 1
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0]) # group the first colum of the array by 60
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train) # change the type of array

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # transform in cube of 927,60,1

# ADD = add layer in the model
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2) # train the model here

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs) # scale between 0 / 1 

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test) # natrix of 248,60

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # transform in cube of 248,60,1
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


#################################
# result 
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

#for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
# plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])


####################################
# predict next day data

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs) # scale between 0 / 1 

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test) # natrix of 248,60

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # transform in cube of 248,60,1
futur_price = model.predict(X_test)
futur_price = scaler.inverse_transform(futur_price)