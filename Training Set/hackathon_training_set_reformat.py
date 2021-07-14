import pandas as pd
import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'

di = pd.read_csv(ddir + 'trainingset_labeled.csv')

#One-Hot Encoding
y = pd.get_dummies(di.labels)
di_oh = di.join(y)
di_oh = di_oh.drop('labels', axis = 1)

# Features and Targets
features = di_oh[['x','y','z']].to_numpy().tolist()
targets = di_oh[list(y.columns)].to_numpy().tolist()

