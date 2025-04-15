import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("Atlanta_Wind.csv")

df['Observation_Date'] = pd.to_datetime(df['Observation_Date'])
df['Day_of_Year'] = df['Observation_Date'].dt.dayofyear

# Features and target
features = ['Wind_Velocity_Avg', 'Wind_Direction_Avg', 'Wind_Velocity_Max', 'Wind_Velocity_Min', 'Day_of_Year']
target = 'Wind_Velocity_Avg'

# Normalize data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Create sequences
def create_sequences(data, target_column, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
        labels.append(data.iloc[i + seq_length][target_column])
    return np.array(sequences), np.array(labels)

sequence_length = 10
X, y = create_sequences(df_scaled, 'Wind_Velocity_Avg', sequence_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Define optimized LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(75, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(50, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1)

# Evaluate model
val_mse = model.evaluate(X_val, y_val)
rmse = np.sqrt(val_mse)
print(f"Validation MSE: {val_mse}")
print(f"Validation RMSE: {rmse}")

# Plot training history
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# Predict and compare
predictions = model.predict(X_val)
plt.figure(figsize=(10,5))
plt.plot(y_val, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Actual vs Predicted Wind Velocity")
plt.legend()
plt.savefig("prediction_comparison.png")
plt.show()

# Scatter plot of actual vs predicted values
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_val, y=predictions.flatten())
plt.xlabel("Actual Wind Velocity")
plt.ylabel("Predicted Wind Velocity")
plt.title("Scatter Plot: Actual vs Predicted Wind Velocity")
plt.savefig("scatter_plot.png")
plt.show()
