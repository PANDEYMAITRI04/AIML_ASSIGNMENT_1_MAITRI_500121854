import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
data = pd.read_csv(r'C:\Users\CHARVI GANJOO\Downloads\temperature.csv')  # Replace with your file path

# Check the column names and first few rows
print("Columns in the dataset:", data.columns)
print("First few rows of the dataset:\n", data.head())

# Clean the data to remove any non-numeric characters
for col in data.columns[1:]:  # Assuming the first column is the region name
    data[col] = data[col].replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters

# Convert the cleaned data to numeric (if not already)
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 2. Extract Bihar's Data (assuming Bihar is the first row in the dataset)
bihar_data = data.iloc[0, 1:]  # Extract the first row, excluding the region name (first column)

# Print Bihar's data to confirm extraction
print("Bihar Data (Temperature):\n", bihar_data)

# 3. Impute missing values for Bihar (if any)
imputer = KNNImputer(n_neighbors=5)
bihar_data_imputed = imputer.fit_transform(bihar_data.values.reshape(1, -1))

# Convert imputed data back to DataFrame
bihar_data_imputed = pd.DataFrame(bihar_data_imputed, columns=bihar_data.index)

# Print the imputed data for Bihar
print("Imputed Bihar Data:\n", bihar_data_imputed)

# 4. Normalize/Scale the Data (apply scaling after imputing)
scaler = StandardScaler()
scaled_data = data.copy()  # Create a copy of the original dataset
scaled_data.iloc[:, 1:] = scaler.fit_transform(scaled_data.iloc[:, 1:])

# Print the scaled dataset
print("Scaled Data:\n", scaled_data)

# 5. Prepare data for training
# Exclude the non-numeric 'region' column for model training
X = scaled_data.iloc[:, 2:]  # Only numerical columns, excluding region name and target
y = scaled_data['Mar']  # Target variable (March temperature)

# 6. Fit a RandomForestRegressor to predict the temperature
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X, y)

# 7. Make predictions and evaluate the model
y_pred = regressor.predict(X)

# Evaluation
print("Regression Model Evaluation:")
print("MSE:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# 8. Example of making a prediction with the model
sample_bihar_input = scaled_data.iloc[0, 2:].values.reshape(1, -1)  # Features only, no region or target
predicted_temp_scaled = regressor.predict(sample_bihar_input)

# Inverse transform the scaled prediction back to the original temperature scale
predicted_temp_original = scaler.inverse_transform(
    np.hstack([[[0]], [[0]], predicted_temp_scaled.reshape(-1, 1), np.zeros((1, X.shape[1] - 2))])
)[0, 2]

print("Predicted March Temperature for Bihar (Original Scale):", predicted_temp_original)
