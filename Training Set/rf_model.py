import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import GridSearchCV

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'rf_trainingset.csv')

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

y_test_oh = np.array(pd.get_dummies(y_test))
y_pred_oh = np.array(pd.get_dummies(y_pred))

for i in range(n_classes):
    roc_auc = roc_auc_score(y_test_oh[:, i], y_pred_oh[:, i])
    print(i, ' : ', roc_auc)
