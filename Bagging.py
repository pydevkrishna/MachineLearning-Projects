import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a sample dataset of houses
# In real life, you would load your own data
np.random.seed(42)  # for reproducibility

# Generate sample data
n_samples = 1000
data = {
    'size': np.random.uniform(500, 5000, n_samples),  # Square footage
    'bedrooms': np.random.randint(1, 6, n_samples),   # Number of bedrooms
    'bathrooms': np.random.randint(1, 4, n_samples),  # Number of bathrooms
    'age': np.random.randint(0, 50, n_samples),       # Age of the house
    'price': np.zeros(n_samples)                      # Price (to be calculated)
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate price based on features (with some noise)
df['price'] = (
    df['size'] * 100 + 
    df['bedrooms'] * 50000 + 
    df['bathrooms'] * 30000 - 
    df['age'] * 1000 + 
    np.random.normal(0, 50000, n_samples)
)

# Step 2: Prepare features (X) and target (y)
X = df[['size', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the Bagging model
base_model = DecisionTreeRegressor(random_state=42)
bagging_model = BaggingRegressor(
    base_estimator=base_model,
    n_estimators=100,
    random_state=42
)
bagging_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = bagging_model.predict(X_test_scaled)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show(block=False)
plt.pause(0.1)

# Step 9: Visualize feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Size', 'Bedrooms', 'Bathrooms', 'Age'],
    'Importance': np.mean([tree.feature_importances_ for tree in bagging_model.estimators_], axis=0)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance in House Price Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show(block=False)
plt.pause(0.1)

# Step 10: Make predictions for new houses
new_houses = pd.DataFrame({
    'size': [2000, 3000, 1500, 4000],
    'bedrooms': [3, 4, 2, 5],
    'bathrooms': [2, 3, 1, 4],
    'age': [5, 10, 20, 2]
})

# Scale the new houses
new_houses_scaled = scaler.transform(new_houses)

# Make predictions
predictions = bagging_model.predict(new_houses_scaled)

print("\nPredictions for new houses:")
for i, (size, bedrooms, bathrooms, age) in enumerate(zip(
    new_houses['size'], new_houses['bedrooms'], 
    new_houses['bathrooms'], new_houses['age']
)):
    print(f"\nHouse {i+1}:")
    print(f"Size: {size:.0f} sq ft")
    print(f"Bedrooms: {bedrooms}")
    print(f"Bathrooms: {bathrooms}")
    print(f"Age: {age} years")
    print(f"Predicted Price: ${predictions[i]:,.2f}")

# Step 11: Interactive prediction system
def predict_house():
    print("\nHouse Price Prediction System")
    print("----------------------------")
    
    while True:
        try:
            size = float(input("\nEnter house size (sq ft): "))
            bedrooms = int(input("Enter number of bedrooms: "))
            bathrooms = int(input("Enter number of bathrooms: "))
            age = int(input("Enter house age (years): "))
            
            if size > 0 and bedrooms > 0 and bathrooms > 0 and age >= 0:
                # Create and scale the house
                house = pd.DataFrame({
                    'size': [size],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'age': [age]
                })
                house_scaled = scaler.transform(house)
                
                # Make prediction
                prediction = bagging_model.predict(house_scaled)[0]
                
                print(f"\nPredicted Price: ${prediction:,.2f}")
            else:
                print("Please enter valid values!")
                
        except ValueError:
            print("Please enter valid numbers!")
            
        if input("\nCheck another house? (y/n): ").lower() != 'y':
            plt.close('all')
            break

# Run the interactive prediction system
predict_house()
