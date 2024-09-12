import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for rainfall (in mm) and water level (in meters)
n_samples = 1000
rainfall = np.random.uniform(0, 300, n_samples)  # Rainfall between 0 and 300 mm
water_level = np.random.uniform(0, 10, n_samples)  # Water level between 0 and 10 meters

# Define flood probability based on conditions
def flood_probability(rain, water):
    return 1 if rain > 150 and water > 5 else 0

flood_prob = np.array([flood_probability(rain, water) for rain, water in zip(rainfall, water_level)])

# Create a DataFrame
data = pd.DataFrame({
    'Rainfall (mm)': rainfall,
    'Water Level (m)': water_level,
    'Flood Probability': flood_prob
})

# Save the DataFrame to a CSV file in the current directory
data.to_csv('flood_prediction_data.csv', index=False)

print("CSV file saved successfully to your local folder!")
