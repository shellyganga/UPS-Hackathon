import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pickle
import xgboost as xgb

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'trainingset_labeled.csv')

disub = di[['x','y','z','labels']]
disub = disub.replace([2.,3.,4.,5.,6.,7.], 1.)

train, test = train_test_split(disub, test_size=0.2, random_state=42, shuffle=True)

X_train = train.copy()
y_train = X_train.pop('labels')

X_test = test.copy()
y_test = X_test.pop('labels')

bst = xgb.XGBClassifier(objective='binary:logistic',
                  base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.5, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.03, max_delta_step=0.1, max_depth=3,
                  min_child_weight=0, missing=np.nan, monotone_constraints='()',
                  n_estimators=5, n_jobs=8, num_parallel_tree=1, random_state=0,
                  reg_alpha=0.1, reg_lambda=0.01, scale_pos_weight=2, subsample=0.2,
                  tree_method='exact', use_label_encoder=False,
                  validate_parameters=1, verbosity=None )

bst.fit(X_train, y_train)

pickle.dump(bst, open('thepickledmodel.pkl','wb'))