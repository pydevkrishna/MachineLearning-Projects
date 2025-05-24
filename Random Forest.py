import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a sample dataset of credit card transactions
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

# Step 4: Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 8: Visualize feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Amount', 'Time', 'Location'],
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance in Fraud Detection')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Step 9: Make predictions for new transactions
new_transactions = pd.DataFrame({
    'amount': [500, 50, 1000, 75],
    'time': [2, 14, 23, 9],
    'location': [1, 3, 4, 2]
})

predictions = rf_model.predict(new_transactions)
probabilities = rf_model.predict_proba(new_transactions)

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

# Step 10: Interactive prediction system
def predict_transaction():
    print("\nCredit Card Fraud Detection System")
    print("----------------------------------")
    
    while True:
        try:
            amount = float(input("\nEnter transaction amount: $"))
            time = float(input("Enter time of transaction (0-24): "))
            location = int(input("Enter location (1-4): "))
            
            if 0 <= time <= 24 and 1 <= location <= 4:
                # Create and predict
                transaction = pd.DataFrame({
                    'amount': [amount],
                    'time': [time],
                    'location': [location]
                })
                
                prediction = rf_model.predict(transaction)[0]
                probability = rf_model.predict_proba(transaction)[0][1]
                
                print(f"\nPrediction: {'Fraud' if prediction == 1 else 'Legitimate'}")
                print(f"Fraud Probability: {probability:.2%}")
            else:
                print("Please enter valid values!")
                
        except ValueError:
            print("Please enter valid numbers!")
            
        if input("\nCheck another transaction? (y/n): ").lower() != 'y':
            break

# Uncomment the line below to run the interactive prediction system
# predict_transaction()
