import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/shellyganga/UPS-Hackathon-Resources/main/Training%20Set/trainingset_labeled.csv?token=AHT6SSFMJUJ73WF6PSCLIQ3A7B6BG")
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

#from keras.preprocessing.sequence import TimeseriesGenerator
import keras

di = pd.read_csv('https://raw.githubusercontent.com/shellyganga/UPS-Hackathon-Resources/main/Training%20Set/trainingset_labeled.csv?token=AHT6SSDYAEP53ZDJSWO467DA7CWLE')
#di['uptimemilli_diff'] = (di['uptimeNanos'] - np.full((di.shape[0],),di['uptimeNanos'].min()))*1e-6
train, test = train_test_split(di, test_size=0.3, random_state=42, shuffle=True)

di['labels'] = di['labels'].fillna(7.)

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
TIME_STEPS = 3068
STEP = 50

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

# df = pd.read_csv('https://raw.githubusercontent.com/jair-jr/driverBehaviorDataset/master/data/16/acelerometro_terra.csv')
# print(df.head())
# df = df[21:]
# new_df = df.groupby('timestamp')
#
# thing = df["timestamp"].get_group('14/05/2016 10:54:34')
# print(len(thing))
# arr = df['timestamp'].unique()
# print(arr)
# for elem in arr:
#     new_df = df.loc[df['timestamp'] == elem]
#     print(len(new_df))
#     if(len(new_df)!=50):
#         print(elem)
# print(df['timestamp'].tolist())
# df = df[df['timestamp'] == "14/05/2016 10:54:33"]
#print(new_df)
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# arr = df["timestamp"].unique()
# print(arr[0])
# print(arr)
# for elem in arr:
#     new_df = df.groupby(by=[elem])
#     print(len(new_df))