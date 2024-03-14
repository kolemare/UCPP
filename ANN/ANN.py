import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocessor import DataVisualizer

# Load and preprocess the dataset
visualizer = DataVisualizer('../udataset/vehicles.csv')
df1 = visualizer.load_and_describe_data()
columns_to_remove = ['id', 'url', 'region_url', 'cylinders', 'title_status', 'VIN', 'size', 'paint_color', 'image_url', 'description', 'county']
df1 = df1.drop(columns=columns_to_remove)
df1 = df1.dropna()

# Splitting the dataset into features and target variable
X = df1.drop(columns=['price']).values
y = df1['price'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the third hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the fourth hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))  # No activation function for regression

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=64, epochs=100)

# Part 4 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)

# Evaluating the model using metrics such as MAE, MSE, or RMSE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')