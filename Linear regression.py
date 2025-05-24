import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load and prepare the data
# We'll use a simple dataset of house prices
# You can replace this with your own data
data = {
    'size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}
df = pd.DataFrame(data)

# Step 2: Prepare features (X) and target (y)
X = df[['size']]  # Feature: house size
y = df['price']   # Target: house price

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")

# Step 7: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.show()

# Step 8: Make a prediction for a new house
new_house_size = 2000
predicted_price = model.predict([[new_house_size]])
print(f"\nPredicted price for a {new_house_size} sq ft house: ${predicted_price[0]:,.2f}")
