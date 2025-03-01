import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# importing Data
filePath = "/Users/agamjotsandhu/Desktop/Learning Data science/credit card fraud detection/creditcard.csv"
cardData = pd.read_csv(filePath)

# Understanding the data set
print(f"Head of the data: \n {cardData.head()} \n")
print(f"Tail of the data: \n {cardData.tail()}\n")
print(f"Data information: \n {cardData.info()}\n")
print(f"Checking null values: \n {cardData.isnull().sum()}\n")
print(f"Checking data balance: {cardData["Class"].value_counts()}\n")

# splitting into fraudulent and legit transactions
fraud = cardData[cardData.Class == 1]
legit = cardData[cardData.Class == 0]

# understanding any differences between fraudulent and legit transactions
print(legit.Amount.describe())
print(fraud.Amount.describe())
print((cardData.groupby('Class').mean()))
# getting a visual of both types
plt.boxplot([legit.Amount, fraud.Amount], showfliers = False)
plt.show()

# taking a random sample of the legit transactions to make the data set balanced
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# dividing data set into explanatory and response variables
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


models = [LogisticRegression(random_state=1), RandomForestClassifier(random_state=1)]
model_names = ["Logistic Regression", "Random Forest Classifier"]

for i in range(2):
    #training the model 
    model = models[i]
    model.fit(X_train, Y_train)

    # testing how accurate the model is
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print(f'{model_names[i]} accuracy on Training data : {training_data_accuracy}')

    # accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)  
    print(f'{model_names[i]} accuracy score on Test Data : {test_data_accuracy} \n')
