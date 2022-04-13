# Libraries required for Facebook Stock Value Prediction using an LSTM Neural Network
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Reading CSV File and dropping any column that is not Facebook
df = pd.read_csv('/Users/jack/Desktop/social media stocks 2012-2022.csv')
df.drop(df.index[df['Symbol'] != 'FB'], inplace=True)

# Show our data and see other attributes like dimensions, etc
print(df.head())
print(df.shape)

# Visualize closing price history before any LSTM
plt.figure(figsize=(16, 8))
plt.title('Close Price History (For Facebook)')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($USD)', fontsize=18)
plt.show()

# Now we'll create a new dataframe with only the Close Column, and then
# Convert to a Numpy Array, so we can train the LSTM Model
data = df.filter(['Close'])
dataset = data.values

# train our data on 80% of the close values
training_data = math.ceil(len(dataset) * 0.8)

# Now we will scale our data, this is important because it helps reduce the data
# in terms of the same thing (prevents comparing apples to oranges)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)  # - Finds the min/max values for scaling and transforms the data
# based on these

# Now we'll create our training dataset - What we are basing our LSTM model off of
train_data = scaled_data[0:training_data, :]

# Now we split our data into x and y training sets
x_train = []  # explanatory variate
y_train = []  # response variate

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays, and reshape data into a 3-D shape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Now we will build our LSTM Model with 126 Neurons and our x_train numpy array
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile model now using the adam optimizer as well as our mean squared error as a way of calculating best fit
model.compile(optimizer='adam', loss='mean_squared_error')

# Train our model now (Note 'fit' is another name for train)
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create new array containing scaled values
test_data = scaled_data[training_data - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data:, :]  # - values we want our model to predict

# Get our x_test values
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert data to a numpy array now, and get predictions
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # - unscales values to predictions

# Now we'll evaluate the model
# We will do this by getting the Root Mean Squared Error i.e. standard deviation of standardized residuals
# Note that a value of 0 means the model predicted the model exactly
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)  # - 21.81847776109784

# Plot our data + Visualize the data & Prediction
train = data[:training_data]
valid = data[training_data:]
valid['Predictions'] = predictions
plt.figure(figsize=(16, 8))
plt.title('LSTM Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($USD)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
