# Credit-Card-Fraud-Detection
Overview
This project focuses on detecting fraudulent credit card transactions using machine learning models. Given the highly imbalanced nature of fraud detection datasets, the approach ensures proper data balancing and model evaluation to achieve high accuracy. This is based on a kaggle data set

Methodology

Data Preprocessing
  The dataset was loaded and explored for missing values and class imbalances.
  Fraudulent and legitimate transactions were separated for further analysis.
  To balance the dataset, a random sample of legitimate transactions was selected to match the number of fraudulent transactions.

Feature Engineering & Splitting
  The dataset was divided into explanatory variables (X) and the response variable (Y).
  Data was split into training and test sets (80%-20%) with stratification to preserve class balance.
  
Model Selection & Training
  Four machine learning models were implemented:
    Logistic Regression
    Random Forest Classifier
    XGBoost
    Neural Networks
  Each model was trained on the balanced dataset.

Evaluation
  Model performance was assessed using accuracy scores on both the training and test sets.
  The best-performing model achieved 93% accuracy on the test data.

Results
  Logistic Regression Accuracy
    Training: 94%
    Test: 93%

  Random Forest Classifier Accuracy
    Training: 100%
    Test: 93%

Conclusion
  This project highlights the importance of handling imbalanced datasets in fraud detection. The Random Forest model demonstrated strong      performance, providing an effective solution for detecting fraudulent transactions.
