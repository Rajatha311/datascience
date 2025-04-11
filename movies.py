import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('movies.csv')

print("First few rows of the dataset:")
print(df.head(10))

print("\nMissing values in each column:")
print(df.isna().sum())

print("\nData types of the columns:")
print(df.dtypes)

df['Year'] = df['Year'].replace(r'\(.*\)', '', regex=True)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df['Votes'] = df['Votes'].replace(r'[^\d]', '', regex=True)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

print("\nMissing values after cleaning:")
print(df[['Year', 'Votes', 'Duration', 'Rating']].isna().sum())

df_cleaned = df.dropna(subset=['Year', 'Votes', 'Duration', 'Rating'])

print("\nShape of dataset after dropping rows with missing values:")
print(df_cleaned.shape)

print("\nData types after cleaning:")
print(df_cleaned.dtypes)

X = df_cleaned[['Year', 'Votes', 'Duration']]
y = df_cleaned['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShapes of training and test sets:")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
print("\nLinear Regression Model - Mean Squared Error:", lr_mse)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print("\nRandom Forest Regressor Model - Mean Squared Error:", rf_mse)
