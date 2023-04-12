
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
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
import csv
import requests

csClassfTrain = "train.csv"
csClassfTest = "test.csv"
app_Rec = "application_record.csv"
creditRec = "credit_record.csv"

df = pd.read_csv(csClassfTrain)

print(df.head())

print(df.describe())
