import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import pickle
import xgboost as xgb

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'trainingset_labeled.csv')

# Data Processing
disub = di[['x','y','z','labels']]
disub = disub.replace([2.,3.,4.,5.,6.,7.], 1.)

# Train Test Split
train, test = train_test_split(disub, test_size=0.2, random_state=42, shuffle=True)

X_train = train.copy()
y_train = X_train.pop('labels')

X_test = test.copy()
y_test = X_test.pop('labels')

# Importing Pickled Model
bst = pickle.load(open('thepickledmodel.pkl','rb'))

# Evaluation
y_pred = bst.predict(X_test)

print(y_pred)
        
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

sensitivity = tp/(tp + fn)
specificity = tn/(tn + fp)
accuracy = (tp + tn)/y_test.shape[0]

auc = roc_auc_score(y_test, y_pred)