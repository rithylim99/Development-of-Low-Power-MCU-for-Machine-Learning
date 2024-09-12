import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('../data_gen/flood_prediction_data.csv')

# Prepare the dataset
X = data[['Rainfall (mm)', 'Water Level (m)']].values
y = data['Flood Probability'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model for flood prediction
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),  # 2 input features
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output is flood probability (0 or 1)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Evaluate the model
model.evaluate(X_test, y_test)
# # Convert the model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the model to a .tflite file
# with open('flood_prediction_model.tflite', 'wb') as f:
#     f.write(tflite_model)
# # Apply post-training quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# quantized_model = converter.convert()
#
# # Save the quantized model
# with open('flood_prediction_model_quantized.tflite', 'wb') as f:
#     f.write(quantized_model)
