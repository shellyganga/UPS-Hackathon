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
import tensorflow as tf
#from keras.preprocessing.sequence import TimeseriesGenerator
import keras

di = pd.read_csv('https://raw.githubusercontent.com/shellyganga/UPS-Hackathon-Resources/main/Training%20Set/trainingset_labeled.csv?token=AHT6SSDYAEP53ZDJSWO467DA7CWLE')
di = di[di['labels'] != 7]

#di['uptimemilli_diff'] = (di['uptimeNanos'] - np.full((di.shape[0],),di['uptimeNanos'].min()))*1e-6
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train, test = train_test_split(di, test_size=0.2, random_state=42, shuffle=True)
sns.countplot(x = 'labels',
              data = train,
              order = df.labels.value_counts().index);
plt.show()
sns.countplot(x = 'labels',
              data = test,
              order = df.labels.value_counts().index);
plt.show()
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
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
def plot_cm(y_true, y_pred, class_names):
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(18, 16))
  ax = sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap=sns.diverging_palette(220, 20, n=7),
      ax=ax
  )

  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  plt.show() # ta-da!
plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)
model.save('/Users/shellyschwartz/PycharmProjects/upsHackathon/UPS-Hackathon-Resources/Aggressive_Driving/my_model2')
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