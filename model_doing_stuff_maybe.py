from tensorflow import keras

list = [[[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0],[ 10 ,100 , 0], [0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005],[0 ,1000 , 3005]]]
list2 = [[[1 ,1 , 1]]]

model = keras.models.load_model("/Users/shellyschwartz/PycharmProjects/upsHackathon/UPS-Hackathon-Resources/Aggressive_Driving/my_model")
prediction = model.predict_classes(list)
print(prediction)