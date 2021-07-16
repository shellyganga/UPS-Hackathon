import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

ddir = 'C:/users/spwiz/Downloads/driverBehaviorDataset-master/driverBehaviorDataset-master/data/'

drivers = ['16/','17/','20/','21/']

files = ['aceleracaoLinear_terra.csv','acelerometro_terra.csv','giroscopio_terra.csv']


for d in drivers:
    uptime = pd.DataFrame()
    for f in files:
        di = pd.read_csv(ddir + d + f)
        uptime[f] = di['uptimeNanos']
        print(di.shape[0])
        print(di.columns)

def newcsv(dr, filename):
    di16_acc = pd.read_csv(ddir + dr + 'acelerometro_terra.csv')
    di16_acc = di16_acc.rename(columns = {'x':'accel_x','y':'accel_y','z':'accel_z'})
    
    di16_linacc = pd.read_csv(ddir + dr + 'aceleracaoLinear_terra.csv')
    di16_linacc = di16_linacc.rename(columns = {'x':'linaccel_x','y':'linaccel_y','z':'linaccel_z'})
    di16_linacc = di16_linacc.drop(columns = ['timestamp','uptimeNanos'])
    
    di16_gyro = pd.read_csv(ddir + dr + 'giroscopio_terra.csv')
    di16_gyro = di16_gyro.rename(columns = {'x':'gyro_x','y':'gyro_y','z':'gyro_z'})
    di16_gyro = di16_gyro.drop(columns = ['timestamp','uptimeNanos'])
    
    di16 = pd.concat([di16_acc,di16_linacc,di16_gyro], axis = 1)
    
    di16.to_csv(ddir + filename, index = False)
    
    print(di16.columns)
    
    return 0

filenames = ['16combined.csv','17combined.csv','20combined.csv','21combined.csv']

i = 0
for i in range(4):
    z = newcsv(drivers[i], filenames[i])
    i += 1
