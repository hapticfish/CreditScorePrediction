
"""
CREDIT SCORE PREDICTION APPLICATION

Authors:
Xiong Bee
Quinlan John
Olson Michael
Sigwanz Nicholas

Credits:
https://www.kaggle.com/datasets/parisrohan/credit-score-classification?resource=download
https://www.kaggle.com/code/gopidurgaprasad/amex-credit-score-model/notebook
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction?select=credit_record.csv

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier



csClassfTrain = "train.csv"
csClassfTest = "test.csv"
app_Rec = "application_record.csv"
creditRec = "credit_record.csv"

df = pd.read_csv(csClassfTrain)

df['Credit_Score'] = pd.Categorical(df['Credit_Score'], categories=['poor', 'standard', 'good'])
encoder = OneHotEncoder()
score_encoded = encoder.fit_transform(df[['Credit_Score']]).toarray()

# encoded ScoreData
data_encoded = pd.concat([df.drop(['Credit_Score'], axis=1), pd.DataFrame(score_encoded)], axis=1)



print("this is the encoded data")
print(data_encoded.head())


pd.set_option('display.max_columns', None)

print(df.head(10))

print(df.describe())

#unique values for Credit_Score
# unique_scores = df['Credit_Score'].unique()
# print("Unique values for Credit_Score Columb " +unique_scores)


#features (X) and target (y)
X = data_encoded.drop(['Score_poor', 'Score_standard', 'Score_good'], axis=1)
y = data_encoded[['Score_poor', 'Score_standard', 'Score_good']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_active_cards = int(input("Enter the number of active credit cards: "))
outstanding_debt = float(input("Enter the current outstanding debt: "))
credit_utilization_ratio = float(input("Enter the credit utilization ratio: "))
num_delayed_payments = int(input("Enter the number of delayed payments: "))
num_credit_inquiries = int(input("Enter the number of credit inquiries: "))

user_input = pd.DataFrame({
    'Number of Active Credit Cards': [num_active_cards],
    'Current Outstanding Debt': [outstanding_debt],
    'Credit Utilization Ratio': [credit_utilization_ratio],
    'Number of Delayed Payments': [num_delayed_payments],
    'Number of Credit Inquiries': [num_credit_inquiries]
})

# train random forest classifier on input data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

prediction = rf.predict(user_input)

# convert the predictions back to categorical values
y_pred_cat = pd.DataFrame(encoder.inverse_transform(prediction), columns=['Credit_Score'])

accuracy = accuracy_score(data_encoded.loc[X_test.index, 'Score'], y_pred_cat['Score'])
print('Accuracy:', accuracy)

print("Predicted Credit_Score: {}".format(prediction[0]))