# -*- coding: utf-8 -*-
"""MLP_bayesien_opt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/145eJx-iXY6gihXLj1OAYa-CgYhm8jGZd
"""

!pip install bayesian-optimization

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import numpy as np


Excel = "stored.xlsx"
data = pd.read_excel(Excel, sheet_name='Data_ML')
#Drop some features
data=data.drop(columns=['Site', 'Month'])

expected_columns = ['efficiency', 'Cost $US/kg', 'daily yield kg/d']
categorical_features = [ 'Type','Material']
# categorical_features = [ 'Site', 'Month','Type','Material']# Or take all features
numerical_features = [col for col in data.columns if col not in categorical_features + expected_columns]


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


X = data.drop(columns=expected_columns, errors='ignore')
y = data[expected_columns].dropna()


X = preprocessor.fit_transform(X)
y = y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(hidden_units, learning_rate, dropout_rate):
    model = Sequential([
        Dense(int(hidden_units), input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(dropout_rate),
        Dense(int(hidden_units / 2), activation='relu'),
        Dropout(dropout_rate),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])
    return model

def objective(hidden_units, learning_rate, dropout_rate, batch_size, epochs):
    hidden_units = int(hidden_units)
    batch_size = int(batch_size)
    epochs = int(epochs)

    model = create_model(hidden_units, learning_rate, dropout_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return -test_rmse

# Defining hyperparameters
pbounds = {
    'hidden_units': (32, 256),
    'learning_rate': (0.0001, 0.1),
    'dropout_rate': (0.1, 0.5),
    'batch_size': (16, 64),
    'epochs': (50, 200)
}


optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42, verbose=2)

# Hyperparameter optimisation
optimizer.maximize(init_points=5, n_iter=15)


best_params = optimizer.max['params']
best_params['hidden_units'] = int(best_params['hidden_units'])
best_params['batch_size'] = int(best_params['batch_size'])
best_params['epochs'] = int(best_params['epochs'])

print("\nMeilleurs hyperparamètres trouvés :", best_params)

# Model Training
best_model = create_model(
    hidden_units=best_params['hidden_units'],
    learning_rate=best_params['learning_rate'],
    dropout_rate=best_params['dropout_rate']
)

history = best_model.fit(
    X_train, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_data=(X_test, y_test),
    verbose=2
)
# Prediction
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"✅ Final Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"✅ Final Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
print("The best ANN model:")
print(f"📌 Number of layers: 3 (excluding input and output layers)")
print(f"🔹 1st hidden layer: {best_params['hidden_units']} neurons, Activation: ReLU")
print(f"🔹 2nd hidden layer: {best_params['hidden_units'] // 2} neurons, Activation: ReLU")
print(f"🔹 Output layer: {y_train.shape[1]} neurons (Regression output)")
print(f"🔹 Dropout rate: {best_params['dropout_rate']}")
print(f"🔹 Optimizer: Adam, Learning rate: {best_params['learning_rate']}")
print(f"🔹 Batch size: {best_params['batch_size']}, Epochs: {best_params['epochs']}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.legend()
plt.title("Loss curve")
plt.show()

