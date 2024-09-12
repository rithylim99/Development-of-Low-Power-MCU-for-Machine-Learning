import numpy as np
import tensorflow as tf  # Full TensorFlow package

# Load the TensorFlow Lite model from the file
tflite_model_path = "flood_prediction_model_quantized.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# Allocate tensors (required before running inference)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (e.g., rainfall and water level)
rainfall = 200.0  # Example rainfall in mm
water_level = 7.0  # Example water level in meters

# Format the input data as a NumPy array and reshape as required by the model
input_data = np.array([[rainfall, water_level]], dtype=np.float32)

# Set the input tensor with the data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor (flood probability)
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the predicted flood probability
print(f"Predicted flood probability: {output_data[0][0]:.4f}")
