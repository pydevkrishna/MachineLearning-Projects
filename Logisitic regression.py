import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def show_visualization(X_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['amount'], X_test['time'], 
               c=y_test, cmap='coolwarm', alpha=0.6)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Time of Transaction')
    plt.title('Fraud Detection Results\n(Blue: Legitimate, Red: Fraud)')
    plt.colorbar(label='Fraud Status')
    plt.show(block=False)
    plt.pause(0.1)

def main():
    # Step 1: Create sample data
    np.random.seed(42)
    n_samples = 1000
    data = {
        'amount': np.random.uniform(10, 1000, n_samples),
        'time': np.random.uniform(0, 24, n_samples),
        'location': np.random.randint(1, 5, n_samples),
        'is_fraud': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)

    # Step 2: Prepare features and target
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
    show_visualization(X_test, y_test)

    # Step 8: Interactive prediction system
    while True:
        try:
            print("\nEnter transaction details (or 'quit' to exit):")
            amount = float(input("Amount: $"))
            time = float(input("Time (0-24): "))
            location = int(input("Location (1-4): "))
            
            if 0 <= time <= 24 and 1 <= location <= 4:
                transaction = pd.DataFrame({
                    'amount': [amount],
                    'time': [time],
                    'location': [location]
                })
                
                prediction = model.predict(transaction)[0]
                probability = model.predict_proba(transaction)[0][1]
                
                print(f"\nPrediction: {'Fraud' if prediction == 1 else 'Legitimate'}")
                print(f"Fraud Probability: {probability:.2%}")
            else:
                print("Please enter valid values!")
                
        except ValueError:
            print("Please enter valid numbers!")
            
        if input("\nCheck another transaction? (y/n): ").lower() != 'y':
            plt.close()
            break

if __name__ == "__main__":
    main()