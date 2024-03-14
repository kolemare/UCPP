import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessor import DataVisualizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the dataset
visualizer = DataVisualizer('../udataset/vehicles.csv')
df = visualizer.load_and_describe_data()
columns_to_remove = ['id', 'url', 'region_url', 'cylinders', 'title_status', 'VIN', 'size', 'paint_color', 'image_url', 'description', 'county']
df.drop(columns=columns_to_remove, inplace=True)
df.dropna(inplace=True)

# Convert the posting date to a numerical format (seconds since the Unix epoch)
df['posting_date'] = pd.to_datetime(df['posting_date'], utc=True)
df['posting_date'] = (df['posting_date'] - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().astype(float)

# Identify categorical features
categorical_features = ['region', 'model', 'condition', 'manufacturer', 'fuel', 'drive', 'type', 'state', 'transmission']

# Identify numerical features
numerical_features = ['year', 'odometer', 'lat', 'long', 'posting_date']

# Create a ColumnTransformer to apply OneHotEncoder to the categorical features and MinMaxScaler to the numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features),
        ('minmax', MinMaxScaler(), numerical_features)
    ]
)

# Apply the ColumnTransformer to the feature set
X = preprocessor.fit_transform(df.drop(columns=['price']))

# Scale the target variable
y = df['price'].values
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=0)

print("----------------------------")
print(X_train)
print("shape 0 is " + str(X_train.shape[0]))
print("shape 1 is " + str(X_train.shape[1]))
print("----------------------------")

# Convert the sparse matrices to dense arrays
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate

# Adding the third hidden layer
#ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
#ann.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate

# Adding the fourth hidden layer
#ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
#ann.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate


# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))  # No activation function for regression

# Part 3 - Training the ANN

# Compiling the ANN with a lower learning rate
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN on the Training set
ann.fit(X_train_dense, y_train, batch_size=512, epochs=50, shuffle=True)

# Part 4 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test_dense)

# Rescale the predictions back to the original scale
y_pred_rescaled = scaler_y.inverse_transform(y_pred)

# Evaluating the model using metrics such as MAE, MSE, RMSE, and R-squared (R²)
mse = mean_squared_error(y_test, y_pred_rescaled)
mae = mean_absolute_error(y_test, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rescaled)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')