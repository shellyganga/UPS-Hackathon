import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

#from keras.preprocessing.sequence import TimeseriesGenerator
import keras

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'

di = pd.read_csv(ddir + 'trainingset_labeled.csv')


'''
# One-Hot Encoding
y = pd.get_dummies(di.labels)
di_oh = di.join(y)
di_oh = di_oh.drop('labels', axis = 1)

# Features and Targets
features = di_oh[['x','y','z']].to_numpy().tolist()
targets = di_oh[list(y.columns)].to_numpy().tolist()

ts_generator = TimeseriesGenerator(features, targets, length=1, sampling_rate = 1, batch_size = 1)
print(ts_generator[0])
'''

def create_dataset(X, y, time_steps, step):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

# Create Dataset
TIME_STEPS = 50
STEP = 10

X_train, y_train = create_dataset(
    di[['x', 'y', 'z']],
    di.labels,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    di[['x', 'y', 'z']],
    di.labels,
    TIME_STEPS,
    STEP
)

# One Hot Encoding
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

print(X_train.shape, y_train.shape)

# Training Model
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

