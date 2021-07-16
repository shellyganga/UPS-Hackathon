import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


from sklearn.model_selection import GridSearchCV

event_dict = {0:'Non-aggressive Event',1:'Aggressive Right Turn',2:'Aggressive Left Turn',3:'Aggressive Right Lane Change',4:'Aggressive Left Lane Change',5:'Aggressive Acceleration',6:'Aggressive Braking'}

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'rf_trainingset_combined.csv')

factor = pd.factorize(di['labels'])

di['labels'] = factor[0]

train, test = train_test_split(di, test_size=0.3, random_state=42, shuffle=True)

y_train = train.pop('labels')
X_train = train.copy()

y_test = test.pop('labels')
X_test = test.copy()

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)

classifier = RandomForestClassifier(**grid_search.best_params_)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

n_classes = len(list(pd.unique(di['labels'])))

y_test = pd.DataFrame(y_test).replace(event_dict)
y_pred = pd.DataFrame(y_pred).replace(event_dict)

y_test_oh = pd.get_dummies(y_test, prefix = None)
y_pred_oh = pd.get_dummies(y_pred, prefix = None)

def remove_prefix(prefix):
    return lambda x: x[len(prefix):]

y_test_oh = y_test_oh.rename(remove_prefix('labels_'), axis='columns')
y_pred_oh = y_pred_oh.rename(remove_prefix('0_'), axis='columns')

col_test = list(y_test_oh.columns)
col_pred = list(y_pred_oh.columns)
col_not_same = [x for x in col_test if x not in col_pred]

for i in range(n_classes):
    col = col_test[i]
    
    if col not in col_not_same:
        roc_auc = roc_auc_score(y_test_oh[col], y_pred_oh[col])
        print(col, ' : ', roc_auc)

        accuracy = accuracy_score(y_test_oh[col], y_pred_oh[col])
        print(col, ' : ', accuracy)


