import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r'C:\Users\prana\PycharmProjects\MachineLearning\train.csv'
data = pd.read_csv(file_path)

# Select relevant columns for the model
data_filtered = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']]

# Drop rows with missing values
data_filtered = data_filtered.dropna()

# Define features (X) and target (y)
X = data_filtered[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = data_filtered['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the model coefficients and evaluation metrics
model_coefficients = model.coef_
model_intercept = model.intercept_

print("Model Coefficients:", model_coefficients)
print("Model Intercept:", model_intercept)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)