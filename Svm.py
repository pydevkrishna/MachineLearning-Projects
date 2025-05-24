import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Step 2: Prepare features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the SVM model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 9: Visualize feature importance
# Get feature importance using coefficients
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': np.abs(svm_model.coef_[0]) if hasattr(svm_model, 'coef_') else np.zeros(len(data.feature_names))
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot top 10 important features
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
plt.title('Top 10 Important Features')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 10: Make predictions for new cases
# Create sample new cases
new_cases = pd.DataFrame({
    'mean radius': [15.0, 20.0, 12.0],
    'mean texture': [20.0, 25.0, 15.0],
    'mean perimeter': [100.0, 120.0, 80.0],
    'mean area': [800.0, 1000.0, 600.0],
    'mean smoothness': [0.1, 0.15, 0.08],
    'mean compactness': [0.2, 0.3, 0.15],
    'mean concavity': [0.1, 0.2, 0.05],
    'mean concave points': [0.05, 0.1, 0.02],
    'mean symmetry': [0.2, 0.3, 0.15],
    'mean fractal dimension': [0.06, 0.08, 0.04]
})

# Scale the new cases
new_cases_scaled = scaler.transform(new_cases)

# Make predictions
predictions = svm_model.predict(new_cases_scaled)
probabilities = svm_model.predict_proba(new_cases_scaled)

print("\nPredictions for new cases:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nCase {i+1}:")
    print(f"Prediction: {'Malignant' if pred == 0 else 'Benign'}")
    print(f"Malignant Probability: {prob[0]:.2%}")
    print(f"Benign Probability: {prob[1]:.2%}")

# Step 11: Interactive prediction system
def predict_case():
    print("\nBreast Cancer Detection System")
    print("-----------------------------")
    
    while True:
        try:
            # Get input for key features
            radius = float(input("\nEnter mean radius (6-30): "))
            texture = float(input("Enter mean texture (9-40): "))
            perimeter = float(input("Enter mean perimeter (40-190): "))
            area = float(input("Enter mean area (140-2500): "))
            smoothness = float(input("Enter mean smoothness (0.05-0.2): "))
            
            # Create case
            case = pd.DataFrame({
                'mean radius': [radius],
                'mean texture': [texture],
                'mean perimeter': [perimeter],
                'mean area': [area],
                'mean smoothness': [smoothness]
            })
            
            # Add other features with mean values
            for feature in data.feature_names:
                if feature not in case.columns:
                    case[feature] = df[feature].mean()
            
            # Scale the case
            case_scaled = scaler.transform(case)
            
            # Make prediction
            prediction = svm_model.predict(case_scaled)[0]
            probability = svm_model.predict_proba(case_scaled)[0]
            
            print(f"\nPrediction: {'Malignant' if prediction == 0 else 'Benign'}")
            print(f"Malignant Probability: {probability[0]:.2%}")
            print(f"Benign Probability: {probability[1]:.2%}")
            
        except ValueError:
            print("Please enter valid numbers!")
            
        if input("\nCheck another case? (y/n): ").lower() != 'y':
            break

# Uncomment the line below to run the interactive prediction system
# predict_case()
