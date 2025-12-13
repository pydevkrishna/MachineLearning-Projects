import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset of Stock Prices
# In real life, you would use: df = quandl.get("WIKI/GOOGL")
np.random.seed(42)
days = 1000

# Generate synthetic stock data (Random Walk)
dates = pd.date_range(start='2020-01-01', periods=days)
price = 100 + np.cumsum(np.random.randn(days))  # Base price 100
volume = np.random.randint(1000, 10000, days)
high_low_pct = np.abs(np.random.normal(0.02, 0.01, days))  # Volatility
pct_change = np.random.normal(0.00, 0.01, days)            # Daily return

df = pd.DataFrame({
    'Close': price,
    'HL_PCT': high_low_pct,   # High-Low Percentage
    'PCT_Change': pct_change, # Percentage Change
    'Volume': volume
}, index=dates)

# Step 2: Prepare features and labels
# We want to predict the price 'n' days into the future
forecast_out = 30 # Predict 30 days into future
df['Prediction'] = df['Close'].shift(-forecast_out)

# Features (X)
X = np.array(df.drop(['Prediction'], axis=1))
X = X[:-forecast_out] # Remove last 30 days (no target variable)

# Labels (y)
y = np.array(df['Prediction'])
y = y[:-forecast_out] # Remove last 30 days

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
confidence = lr_model.score(X_test_scaled, y_test)
print(f"Model Confidence (R² Score): {confidence:.2%}")

# Step 7: Make predictions for the 'Future' (the last 30 days we excluded)
X_future = np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]
X_future_scaled = scaler.transform(X_future)
forecast_set = lr_model.predict(X_future_scaled)

print("\nPredicted prices for the next 5 days:")
print(forecast_set[:5])

# Step 8: Visualize the results
plt.figure(figsize=(12, 6))
df['Forecast'] = np.nan

# Mapping forecast to dates
last_date = df.iloc[-forecast_out - 1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = pd.to_datetime(next_unix, unit='s')
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    next_unix += one_day

df['Close'].plot(label='Historical Close Price')
df['Forecast'].plot(label='Forecast (30 days)', color='green')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Forecast (Linear Regression)')
plt.show(block=False)
plt.pause(0.1)

# Step 9: Interactive prediction system
def predict_price():
    print("\nStock Price Prediction System")
    print("-----------------------------")
    
    while True:
        try:
            print("\nEnter market parameters:")
            current_price = float(input("Current Close Price: $"))
            volatility = float(input("High-Low Variance % (e.g., 0.02): "))
            daily_change = float(input("Daily Change % (e.g., 0.01): "))
            volume_input = float(input("Trading Volume: "))
            
            # Create input array
            user_data = pd.DataFrame([[current_price, volatility, daily_change, volume_input]],
                                   columns=['Close', 'HL_PCT', 'PCT_Change', 'Volume'])
            
            # Scale input
            user_data_scaled = scaler.transform(user_data)
            
            # Predict
            prediction = lr_model.predict(user_data_scaled)[0]
            
            print(f"\nPredicted Price in {forecast_out} days: ${prediction:.2f}")
            
        except ValueError:
            print("Please enter valid numbers!")
            
        if input("\nPredict another scenario? (y/n): ").lower() != 'y':
            plt.close('all')
            break

# Run the interactive system
if __name__ == "__main__":
    predict_price()
