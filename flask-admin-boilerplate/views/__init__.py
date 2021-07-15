from flask import json, render_template, request, redirect, url_for, session,jsonify
from app import app
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

#Load CSV File
csvFile = pd.read_csv('/Users/auddin431/Desktop/UPS-Hackathon-Resources/flask-admin-boilerplate/views/trainingset_labeled.csv')

# Data Processing
disub = csvFile[['x','y','z','labels']]
disub = disub.replace([2.,3.,4.,5.,6.,7.], 1.)

# Train Test Split
train, test = train_test_split(disub, test_size=0.2, random_state=42, shuffle=True)

X_train = train.copy()
y_train = X_train.pop('labels')

X_test = test.copy()
y_test = X_test.pop('labels')

@app.route('/', methods=["GET"])
def home():

#Load Pickled Model

    model = pickle.load(open('/Users/auddin431/Desktop/UPS-Hackathon-Resources/flask-admin-boilerplate/views/thepickledmodel.pkl','rb'))

# Evaluation
    y_pred = model.predict(X_test)
    test = pd.unique(y_pred)
    return render_template('index.html')
    # if "username" in session:
    #     return render_template('index.html')
    # else:
    #     return render_template('login.html')

@app.route('/driver_2', methods=["GET"])
def driver_2():

    model = pickle.load(open('/Users/auddin431/Desktop/UPS-Hackathon-Resources/flask-admin-boilerplate/views/thepickledmodel.pkl','rb'))

# Evaluation
    y_pred = model.predict(X_test)
    test = pd.unique(y_pred)
    return render_template('driver_2.html')
    # if "username" in session:
    #     return render_template('index.html')
    # else:
    #     return render_template('login.html')

@app.route('/api', methods=["GET", "POST"])
def api():


    model = pickle.load(open('/Users/auddin431/Desktop/UPS-Hackathon-Resources/flask-admin-boilerplate/views/thepickledmodel.pkl','rb'))

# Evaluation
    y_pred = model.predict(X_test)
    test = pd.unique(y_pred)
    message = {'info': int(test[0])}
    return jsonify(message)


# Register new user
# @app.route('/register', methods=["GET", "POST"])
# def register():
#     if request.method == "GET":
#         return render_template("register.html")
#     elif request.method == "POST":
#         registerUser()
#         return redirect(url_for("login"))

#Check if email already exists in the registratiion page
# @app.route('/checkusername', methods=["POST"])
# def check():
#     return checkusername()

# Everything Login (Routes to renderpage, check if username exist and also verifypassword through Jquery AJAX request)
# @app.route('/login', methods=["GET"])
# def login():
#     if request.method == "GET":
#         if "username" not in session:
#             return render_template("login.html")
#         else:
#             return redirect(url_for("home"))


# @app.route('/checkloginusername', methods=["POST"])
# def checkUserlogin():
#     return checkloginusername()

# @app.route('/checkloginpassword', methods=["POST"])
# def checkUserpassword():
#     return checkloginpassword()

#The admin logout
# @app.route('/logout', methods=["GET"])  # URL for logout
# def logout():  # logout function
#     session.pop('username', None)  # remove user session
#     return redirect(url_for("home"))  # redirect to home page with message

#Forgot Password
# @app.route('/forgot-password', methods=["GET"])
# def forgotpassword():
#     return render_template('forgot-password.html')

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
