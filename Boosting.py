import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

# Step 1: Load the digits dataset (a simpler version of MNIST)
# This dataset contains 8x8 images of handwritten digits
digits = load_digits()

# Create DataFrame
df = pd.DataFrame(digits.data)
df['target'] = digits.target

# Step 2: Prepare features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = gb_model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show(block=False)
plt.pause(0.1)

# Step 9: Visualize feature importance
feature_importance = pd.DataFrame({
    'Pixel': range(64),  # 8x8 image = 64 pixels
    'Importance': gb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot top 20 important pixels
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Pixel'][:20], feature_importance['Importance'][:20])
plt.title('Top 20 Important Pixels')
plt.xlabel('Pixel Index')
plt.ylabel('Importance')
plt.show(block=False)
plt.pause(0.1)

# Step 10: Visualize some predictions
def plot_predictions(X, y_true, y_pred, indices):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].reshape(8, 8), cmap='gray')
        plt.title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# Plot 5 random predictions
random_indices = np.random.choice(len(X_test), 5, replace=False)
plot_predictions(X_test_scaled, y_test, y_pred, random_indices)

# Step 11: Make predictions for new images
# Create sample new images (you can replace these with real images)
new_images = X_test_scaled[:5]  # Using first 5 test images as examples

predictions = gb_model.predict(new_images)
probabilities = gb_model.predict_proba(new_images)

print("\nPredictions for new images:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nImage {i+1}:")
    print(f"Prediction: {pred}")
    print(f"Confidence: {prob[pred]:.2%}")

# Step 12: Interactive prediction system
def predict_image():
    print("\nDigit Recognition System")
    print("----------------------")
    
    while True:
        try:
            # Get input for a simple 8x8 image
            print("\nEnter pixel values (0-16) for an 8x8 image:")
            pixels = []
            for i in range(8):
                row = input(f"Row {i+1} (8 values, space-separated): ").split()
                if len(row) != 8:
                    raise ValueError("Each row must have 8 values")
                pixels.extend([float(x) for x in row])
            
            # Create and scale the image
            image = pd.DataFrame([pixels])
            image_scaled = scaler.transform(image)
            
            # Make prediction
            prediction = gb_model.predict(image_scaled)[0]
            probability = gb_model.predict_proba(image_scaled)[0][prediction]
            
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {probability:.2%}")
            
            # Visualize the input image
            plt.figure(figsize=(4, 4))
            plt.imshow(np.array(pixels).reshape(8, 8), cmap='gray')
            plt.title(f'Predicted: {prediction}')
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            
        except ValueError as e:
            print(f"Error: {e}")
            
        if input("\nCheck another image? (y/n): ").lower() != 'y':
            plt.close('all')
            break

# Run the interactive prediction system
predict_image()
