import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras

di = pd.read_csv('C:/Users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/trainingset_labeled.csv')
di = di[di['labels'] != 7]

train, test = train_test_split(di, test_size=0.2, random_state=42, shuffle=True)

def create_dataset(X, y, time_steps, step):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

# Create Dataset
TIME_STEPS = 100
STEP = 1

X_train, y_train = create_dataset(
    train[['x', 'y', 'z']],
    train.labels,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    test[['x', 'y', 'z']],
    test.labels,
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
  loss='mse',
  optimizer='adam',
  metrics=['acc', tf.keras.metrics.MeanSquaredError()]
)

history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=16,
    validation_split=0.2,
    shuffle=False
)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Saving Model
tf.keras.models.save_model(model, 'C:/Users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/For Demo/my_model_fordemo')
