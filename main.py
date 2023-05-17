
"""
CREDIT SCORE PREDICTION APPLICATION

Authors:
Xiong Bee
Quinlan John
Olson Michael
Sigwanz Nicholas

Credits:
https://www.kaggle.com/datasets/parisrohan/credit-score-classification?resource=download

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from google.colab import files

# Load in data
uploaded = files.upload()
df = pd.read_csv('test.csv')
df_clean = df[["Annual_Income", "Num_Credit_Inquiries", "Monthly_Balance", "Outstanding_Debt", "Credit_Mix"]]
df_clean
df_clean.info()

# remove all rolls with none value
df_clean = df_clean.dropna()
df_clean.info()

# check column "Credit_Mix" for unwated value
df_clean["Credit_Mix"].value_counts()

# showing the result
df_clean["Credit_Mix"].value_counts()

# remove unwanted value
df_clean = df_clean[df_clean["Credit_Mix"].str.contains("_") == False]
df_clean.info()

# a function that checks a value to see if it can be converted to a float number
def try_float(value):
    try:
      return float(value)
    except ValueError:
      return "_"

    # iterate through the dataframe to set unwanted value to NAN
    for ind in range(df_clean.shape[0]):
        annual_income = df_clean.iat[ind, 0]
        monthly_balance = df_clean.iat[ind, 2]
        outstanding_debt = df_clean.iat[ind, 3]

        x = try_float(annual_income)
        if x == "_":
            df_clean = df_clean.replace(annual_income, np.nan)

        x = try_float(monthly_balance)
        if x == "_":
            df_clean = df_clean.replace(monthly_balance, np.nan)

        x = try_float(outstanding_debt)
        if x == "_":
            df_clean = df_clean.replace(outstanding_debt, np.nan)


# showing the result
df_clean.info()

# remove all rows that has a none value
df_clean = df_clean.dropna()
df_clean.info()

# change column "Credit_Mix" value to numbers
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_clean["Credit_Mix"] = le.fit_transform(df_clean["Credit_Mix"])

# change all column data type to float64
df_clean = df_clean.astype(float)
df_clean.info()

df_clean

X = df_clean[["Annual_Income", "Num_Credit_Inquiries", "Monthly_Balance", "Outstanding_Debt"]]
Y = df_clean["Credit_Mix"]

# visualizing preprcessed data
# scatter plot showing annual income vs monthly balance
df_clean.plot(x = "Annual_Income", y = "Monthly_Balance",  kind = "scatter")

#scatter plot showing credit inquiries vs debts
df_clean.plot(x = "Num_Credit_Inquiries", y = "Outstanding_Debt",  kind = "scatter")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25, random_state=1)

clf_random = RandomForestClassifier()
clf_random.fit(X_train, Y_train)

predicted_train = clf_random.predict(X_train)
predicted_test = clf_random.predict(X_test)

print("Results for training data with Random Forest")
print(classification_report(Y_train, predicted_train))

print("Results for testing data with Random Forest")
print(classification_report(Y_test, predicted_test))

clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train, Y_train)

predicted_train = clf_ada.predict(X_train)
predicted_test = clf_ada.predict(X_test)

print("Results for training data with AdaBoostClassifier")
print(classification_report(Y_train, predicted_train))

print("Results for testing data with AdaBoostClassifier")
print(classification_report(Y_test, predicted_test))

clf_GradiB = GradientBoostingClassifier()
clf_GradiB.fit(X_train, Y_train)

predicted_train = clf_GradiB.predict(X_train)
predicted_test = clf_GradiB.predict(X_test)

print("Results for training data with GradientBoostingClassifier")
print(classification_report(Y_train, predicted_train))

print("Results for testing data with GradientBoostingClassifier")
print(classification_report(Y_test, predicted_test))

# GRID SEARCH WIth Cross validation for Random forest

clf_rand = RandomForestClassifier(random_state=9090)
param_grid = {
    'n_estimators': [10, 100, 200],
    'max_depth': [None, 10, 20],

}

gs = GridSearchCV(clf_rand, param_grid=param_grid, cv=5)
gs.fit(X_train, Y_train)

print('Best params', gs.best_params_)

print('Best acuracy', gs.best_score_)

# create a new random forest classifier using the best parameters that were
# discovered using the grid search above.
clf = RandomForestClassifier(**gs.best_params_)
clf.fit(X_train, Y_train)

predicted_train = clf_random.predict(X_train)
predicted_test = clf_random.predict(X_test)

print("Results for testing data with Random Forest")
print(classification_report(Y_test, predicted_test))

# this is cross validation only but without hyper parameter tuning
rf = RandomForestClassifier(random_state=9090 )
scores = cross_val_score(rf, X_train, Y_train, cv=5)
print(scores)

from sklearn import metrics
import seaborn as sns
# Create the Confusion Matrix

cnf_matrix = metrics.confusion_matrix(Y_test, predicted_test)

# Visualizing the Confusion Matrix
class_names = [0,1] # Our diagnosis categories

fig, ax = plt.subplots()
# Setting up and visualizing the plot (do not worry about the code below!)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') # Creating heatmap
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual diagnosis')
plt.xlabel('Predicted diagnosis')

