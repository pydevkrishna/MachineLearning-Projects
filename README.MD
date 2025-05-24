# Machine Learning Projects Collection

A collection of machine learning projects demonstrating various algorithms and their applications.

## Table of Contents
1. [Financial Forecast (Linear Regression)](#financial-forecast)
2. [Credit Card Fraud Detection (Logistic Regression)](#credit-card-fraud)
3. [Movie Recommendation System (KNN)](#movie-recommendation)
4. [Breast Cancer Detection (Decision Trees)](#breast-cancer-detection)
5. [Spam Filter (Naive Bayes)](#spam-filter)
6. [Breast Cancer Detection (SVM)](#breast-cancer-svm)
7. [Credit Card Fraud (Random Forest)](#credit-card-fraud-rf)
8. [Image Recognition (Gradient Boosting)](#image-recognition)
9. [House Price Prediction (Bagging)](#house-price-prediction)

## Project Details

### Financial Forecast
**Algorithm**: Linear Regression
- **Purpose**: Predict stock prices
- **Features**: Historical prices, volume, technical indicators
- **Output**: Predicted future prices
- **Key Functions**:
  - `quandl.get_table()`: Fetches financial data
  - `LinearRegression()`: Creates regression model
  - `preprocessing.scale()`: Normalizes features

### Credit Card Fraud
**Algorithm**: Logistic Regression
- **Purpose**: Identify fraudulent transactions
- **Features**: Transaction amount, time, location
- **Output**: Fraud probability
- **Key Functions**:
  - `LogisticRegression()`: Creates classifier
  - `StandardScaler()`: Scales features
  - `confusion_matrix()`: Evaluates model

### Movie Recommendation
**Algorithm**: K-Nearest Neighbors (KNN)
- **Purpose**: Suggest similar movies
- **Features**: Movie characteristics, user preferences
- **Output**: Movie recommendations
- **Key Functions**:
  - `NearestNeighbors()`: Creates KNN model
  - `kneighbors()`: Finds nearest neighbors
  - `CountVectorizer()`: Processes text data

### Breast Cancer Detection
**Algorithm**: Decision Trees
- **Purpose**: Classify cancer cases
- **Features**: Medical measurements
- **Output**: Malignant/Benign classification
- **Key Functions**:
  - `DecisionTreeClassifier()`: Creates tree model
  - `tree.plot_tree()`: Visualizes decision tree
  - `feature_importances_`: Shows feature importance

### Spam Filter
**Algorithm**: Naive Bayes
- **Purpose**: Classify emails as spam/legitimate
- **Features**: Email content, headers
- **Output**: Spam probability
- **Key Functions**:
  - `MultinomialNB()`: Creates Naive Bayes model
  - `CountVectorizer()`: Processes text
  - `predict_proba()`: Provides probabilities

### Breast Cancer Detection (SVM)
**Algorithm**: Support Vector Machine
- **Purpose**: Classify cancer cases
- **Features**: Medical measurements
- **Output**: Malignant/Benign classification
- **Key Functions**:
  - `SVC()`: Creates SVM classifier
  - `StandardScaler()`: Scales features
  - `classification_report()`: Evaluates model

### Credit Card Fraud (Random Forest)
**Algorithm**: Random Forest
- **Purpose**: Identify fraudulent transactions
- **Features**: Transaction details
- **Output**: Fraud probability
- **Key Functions**:
  - `RandomForestClassifier()`: Creates forest model
  - `feature_importances_`: Shows feature importance
  - `predict_proba()`: Provides probabilities

### Image Recognition
**Algorithm**: Gradient Boosting
- **Purpose**: Classify handwritten digits
- **Features**: Pixel values
- **Output**: Digit classification
- **Key Functions**:
  - `GradientBoostingClassifier()`: Creates model
  - `StandardScaler()`: Scales features
  - `plot_predictions()`: Visualizes results

### House Price Prediction
**Algorithm**: Bagging
- **Purpose**: Predict house prices
- **Features**: House characteristics
- **Output**: Predicted price
- **Key Functions**:
  - `BaggingRegressor()`: Creates bagging model
  - `mean_squared_error()`: Evaluates model
  - `r2_score()`: Calculates R-squared

## Common Workflow
1. **Data Preparation**
   - Load/Create data
   - Handle missing values
   - Feature engineering

2. **Model Training**
   - Split data (train/test)
   - Scale features
   - Train model

3. **Evaluation**
   - Calculate metrics
   - Visualize results
   - Make predictions

## Project Selection Guide

### For Prediction Tasks
- **Linear Regression**: Simple numerical predictions
- **Bagging**: Complex numerical predictions

### For Classification Tasks
- **Logistic Regression**: Binary classification
- **Decision Trees**: Multi-class classification
- **Random Forest**: Complex classification

### For Recommendation Tasks
- **KNN**: Similar item recommendations

### For Text Classification
- **Naive Bayes**: Spam detection, text categorization

## Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Choose the appropriate project for your task
2. Follow the common workflow
3. Use the provided functions and models
4. Evaluate and visualize results

## Contributing
Feel free to contribute to this project by:
1. Adding new machine learning projects
2. Improving existing code
3. Adding more documentation
4. Fixing bugs

## License
This project is licensed under the MIT License - see the LICENSE file for details.
