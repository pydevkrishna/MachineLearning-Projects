import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset of transactions
# In real life, you would load your own data
np.random.seed(42)  # for reproducibility

# Generate sample data
n_samples = 1000
data = {
    'amount': np.random.uniform(10, 1000, n_samples),  # Transaction amount
    'time': np.random.uniform(0, 24, n_samples),      # Time of transaction (0-24 hours)
    'location': np.random.randint(1, 5, n_samples),   # Location (1-4)
    'is_fraud': np.random.randint(0, 2, n_samples)    # Fraud label (0: legitimate, 1: fraud)
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Prepare features (X) and target (y)
X = df[['amount', 'time', 'location']]
y = df['is_fraud']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test['amount'], X_test['time'], 
           c=y_test, cmap='coolwarm', alpha=0.6)
plt.xlabel('Transaction Amount')
plt.ylabel('Time of Transaction')
plt.title('Fraud Detection Results\n(Blue: Legitimate, Red: Fraud)')
plt.colorbar(label='Fraud Status')
plt.show()

# Step 8: Make predictions for new transactions
new_transactions = pd.DataFrame({
    'amount': [500, 50, 1000, 75],
    'time': [2, 14, 23, 9],
    'location': [1, 3, 4, 2]
})

predictions = model.predict(new_transactions)
probabilities = model.predict_proba(new_transactions)

print("\nPredictions for new transactions:")
for i, (amount, time, location) in enumerate(zip(new_transactions['amount'], 
                                               new_transactions['time'], 
                                               new_transactions['location'])):
    print(f"\nTransaction {i+1}:")
    print(f"Amount: ${amount:.2f}")
    print(f"Time: {time:.1f} hours")
    print(f"Location: {location}")
    print(f"Prediction: {'Fraud' if predictions[i] == 1 else 'Legitimate'}")
    print(f"Fraud Probability: {probabilities[i][1]:.2%}")
