##### COPY FROM HERE ######

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

di = pd.read_csv('C:/Users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/For Demo/test_fordemo.csv')
model = keras.models.load_model("/Users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/For Demo/my_model_fordemo")

event_dict = {0:'Non-aggressive Event',1:'Aggressive Right Turn',2:'Aggressive Left Turn',3:'Aggressive Right Lane Change',4:'Aggressive Left Lane Change',5:'Aggressive Acceleration',6:'Aggressive Braking'}
event_list = ['Non-aggressive Event','Aggressive Right Turn','Aggressive Left Turn','Aggressive Right Lane Change','Aggressive Left Lane Change','Aggressive Acceleration','Aggressive Braking']

#cols = ['x','y','z']
cols = ['accel_x','accel_y','accel_z','linaccel_x','linaccel_y','linaccel_z','gyro_x','gyro_y','gyro_z']

def demo_data(X, time_steps, step, model):
    XX = X[['x','y','z']]

    Xs = []
    for i in range(0, len(XX) - time_steps, step):
        v = XX.iloc[i:(i + time_steps)].values
        Xs.append(v)
    Xs = np.array(Xs)

    prediction = model.predict_classes(Xs)

    t_inseconds = time_steps * (1/50)

    timestamps = np.arange(0,t_inseconds*prediction.shape[0],t_inseconds)

    prediction_oh = pd.get_dummies(prediction)
    prediction_oh = prediction_oh.rename(columns = event_dict)

    demo_df = pd.DataFrame(np.nan, index = [i for i in range(0,prediction.shape[0])], columns = event_list)
    print(demo_df)

    for c in prediction_oh.columns:
        demo_df[c] = prediction_oh[c]

    demo_df = demo_df.fillna(0)

    demo_df['timestamp'] = timestamps

    return demo_df

# Create Dataset
TIME_STEPS = 10
STEP = 5

'''
drivers = list(pd.unique(di['driver']))

for d in drivers:
    disub = di[di['driver'] == d]

    demo_df  = demo_data(
        di[['x', 'y', 'z']],
        TIME_STEPS,
        STEP,
        model
    )

    df_freq = demo_df.apply(lambda x: round(sum(x)/di.shape[0] * 100 ,1))
    df_freq = df_freq.drop('timestamp')
'''

demo_df  = demo_data(
    di[cols],
    TIME_STEPS,
    STEP,
    model
)

df_freq = demo_df.apply(lambda x: round(sum(x)/di.shape[0] * 100 ,1))
df_freq = df_freq.drop('timestamp')

##### COPY FROM HERE ######


print(demo_df)
demo_df.to_csv('C:/Users/spwiz/Downloads/demo_df.csv', index = True)
