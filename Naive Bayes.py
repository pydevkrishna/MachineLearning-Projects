import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset of emails
# In real life, you would load your own data
emails = [
    "Get rich quick! Click here to win a million dollars!",
    "Meeting tomorrow at 10 AM in the conference room",
    "URGENT: Your account needs verification",
    "Please review the attached report",
    "Congratulations! You've won a free iPhone!",
    "Team meeting notes from yesterday",
    "Your password has been compromised",
    "Project deadline extension",
    "Claim your prize now!",
    "Please submit your timesheet by Friday",
    "Exclusive offer just for you!",
    "New software update available",
    "Your account has been suspended",
    "Monthly team meeting schedule",
    "Free trial offer - limited time only"
]

# Labels: 1 for spam, 0 for legitimate
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Create DataFrame
df = pd.DataFrame({
    'email': emails,
    'is_spam': labels
})

# Step 2: Prepare the data
# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['is_spam']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = nb_model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Visualize the results
# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get feature importance (log probabilities)
feature_importance = pd.DataFrame({
    'Word': feature_names,
    'Spam_Probability': np.exp(nb_model.feature_log_prob_[1]),
    'Legitimate_Probability': np.exp(nb_model.feature_log_prob_[0])
})

# Sort by spam probability
feature_importance = feature_importance.sort_values('Spam_Probability', ascending=False)

# Plot top 10 spam words
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Word'][:10], feature_importance['Spam_Probability'][:10])
plt.title('Top 10 Words Associated with Spam')
plt.xlabel('Words')
plt.ylabel('Spam Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 8: Make predictions for new emails
new_emails = [
    "Please review the project proposal",
    "You've won a free vacation! Click here!",
    "Meeting rescheduled to 2 PM",
    "URGENT: Your account needs immediate attention"
]

# Transform new emails
new_emails_transformed = vectorizer.transform(new_emails)

# Make predictions
predictions = nb_model.predict(new_emails_transformed)
probabilities = nb_model.predict_proba(new_emails_transformed)

print("\nPredictions for new emails:")
for email, pred, prob in zip(new_emails, predictions, probabilities):
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Legitimate'}")
    print(f"Spam Probability: {prob[1]:.2%}")

# Step 9: Interactive spam filter
def check_email():
    print("\nSpam Filter System")
    print("-----------------")
    
    while True:
        email = input("\nEnter an email to check (or 'quit' to exit): ")
        if email.lower() == 'quit':
            break
            
        # Transform the email
        email_transformed = vectorizer.transform([email])
        
        # Make prediction
        prediction = nb_model.predict(email_transformed)[0]
        probability = nb_model.predict_proba(email_transformed)[0][1]
        
        print(f"\nPrediction: {'Spam' if prediction == 1 else 'Legitimate'}")
        print(f"Spam Probability: {probability:.2%}")
        
        # Show top words that influenced the decision
        words = email.lower().split()
        word_probs = []
        for word in words:
            if word in feature_names:
                idx = list(feature_names).index(word)
                word_probs.append((word, feature_importance.iloc[idx]['Spam_Probability']))
        
        if word_probs:
            print("\nKey words that influenced the decision:")
            for word, prob in sorted(word_probs, key=lambda x: x[1], reverse=True):
                print(f"- {word}: {prob:.2%}")

# Uncomment the line below to run the interactive spam filter
# check_email()
