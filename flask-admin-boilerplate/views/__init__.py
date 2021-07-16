from flask import json, render_template, request, redirect, url_for, session,jsonify
from main import app
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pathlib
import json

path = pathlib.Path(__file__).parent.resolve()


di = pd.read_csv(path/'test_fordemo.csv')
csv_read = pd.read_csv(path/'demo_df_artificial.csv')
model = keras.models.load_model(path/"model_basic_version.h5")
print("I am here")

event_dict = {0:'Non-aggressive Event',1:'Aggressive Right Turn',2:'Aggressive Left Turn',3:'Aggressive Right Lane Change',4:'Aggressive Left Lane Change',5:'Aggressive Acceleration',6:'Aggressive Braking'}
event_list = ['Non-aggressive Event','Aggressive Right Turn','Aggressive Left Turn','Aggressive Right Lane Change','Aggressive Left Lane Change','Aggressive Acceleration','Aggressive Braking']
cols = ['x','y','z']
#cols = ['accel_x','accel_y','accel_z','linaccel_x','linaccel_y','linaccel_z','gyro_x','gyro_y','gyro_z']

def demo_data(X, time_steps, step, model):
    XX = X[cols]

    Xs = []
    for i in range(0, len(XX) - time_steps, step):
        v = XX.iloc[i:(i + time_steps)].values
        Xs.append(v)
    Xs = np.array(Xs)

    print(f"{Xs.shape = }")
    prediction = model.predict_classes(Xs)
    

    t_inseconds = time_steps * (1/50)

    timestamps = np.arange(0,t_inseconds*prediction.shape[0],t_inseconds)

    prediction_oh = pd.get_dummies(prediction)
    prediction_oh = prediction_oh.rename(columns = event_dict)

    demo_df = pd.DataFrame(np.nan, index = [i for i in range(0,prediction.shape[0])], columns = event_list)
    #print(demo_df)

    for c in prediction_oh.columns:
        demo_df[c] = prediction_oh[c]

    demo_df = demo_df.fillna(0)

    demo_df['timestamp'] = timestamps

    return demo_df

# Create Dataset #Later will change
TIME_STEPS = 200
STEP = 5

demo_df  = demo_data(
    di[cols],
    TIME_STEPS,
    STEP,
    model
)
jsonOb = demo_df.to_json(path_or_buf=None, orient=None, 
date_format=None, double_precision=10, 
force_ascii=True, 
date_unit='ms', 
default_handler=None, lines=False, 
compression='infer', index=True)


jsonOb1 = csv_read.to_json(path_or_buf=None, orient=None, 
date_format=None, double_precision=10, 
force_ascii=True, 
date_unit='ms', 
default_handler=None, lines=False, 
compression='infer', index=True)


#print(jsonOb)
@app.route('/', methods=["GET"])
def home():

#Load Pickled Model

#     model = pickle.load(open(path/'thepickledmodel.pkl','rb'))

# # Evaluation
#     y_pred = model.predict(X_test)
#     test = pd.unique(y_pred)
    df_freq = csv_read.apply(lambda x: round(sum(x)/csv_read.shape[0] * 100 ,1))
    df_freq = df_freq.drop('timestamp')

    jsonFreqOb = df_freq.to_json(path_or_buf=None, orient=None, 
        date_format=None, double_precision=10, 
        force_ascii=True, 
        date_unit='ms', 
        default_handler=None, lines=False, 
        compression='infer', index=True)
    finalRet = json.loads(jsonFreqOb)
    return render_template('index.html',
        nonAggresive=finalRet["Non-aggressive Event"],
        aggRightTurn=finalRet["Aggressive Right Turn"],
        aggLeftTurn=finalRet["Aggressive Left Turn"],
        aggRightLane=finalRet["Aggressive Right Lane Change"],
        aggLeftLane=finalRet["Aggressive Left Lane Change"],
        aggAccel=finalRet["Aggressive Acceleration"],
        aggBrake=finalRet["Aggressive Braking"]
    )


@app.route('/driver_2', methods=["GET"])
def driver_2():

#     model = pickle.load(open(path/'thepickledmodel.pkl','rb'))

# # Evaluation
#     y_pred = model.predict(X_test)
#     test = pd.unique(y_pred)
    return render_template('driver_2.html')

@app.route('/api', methods=["GET", "POST"])
def api():
#     model = pickle.load(open(path/'thepickledmodel.pkl','rb'))

# # Evaluation
#     y_pred = model.predict(X_test)
#     test = pd.unique(y_pred)
#     message = {'info': int(test[0])}
    return jsonOb1

@app.route('/freqapi',methods=["GET","POST"])
def freqapi():
    return render_template('index.html')
#404 Page
@app.route('/404', methods=["GET"])
def errorpage():
    return render_template("404.html")

#Blank Page
@app.route('/blank', methods=["GET"])
def blank():
    return render_template('blank.html')

#Buttons Page
@app.route('/buttons', methods=["GET"])
def buttons():
    return render_template("buttons.html")

#Cards Page
@app.route('/cards', methods=["GET"])
def cards():
    return render_template('cards.html')

#Charts Page
@app.route('/charts', methods=["GET"])
def charts():
    return render_template("charts.html")

#Tables Page
@app.route('/tables', methods=["GET"])
def tables():
    return render_template("tables.html")

#Utilities-animation
@app.route('/utilities-animation', methods=["GET"])
def utilitiesanimation():
    return render_template("utilities-animation.html")

#Utilities-border
@app.route('/utilities-border', methods=["GET"])
def utilitiesborder():
    return render_template("utilities-border.html")

#Utilities-color
@app.route('/utilities-color', methods=["GET"])
def utilitiescolor():
    return render_template("utilities-color.html")

#utilities-other
@app.route('/utilities-other', methods=["GET"])
def utilitiesother():
    return render_template("utilities-other.html")
