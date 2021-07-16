import pandas as pd
import numpy as np
from scipy import stats

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'trainingset_labeled_combined.csv')

di = di[di['labels'] != 7]

cols = ['accel_x','accel_y','accel_z','linaccel_x','linaccel_y','linaccel_z','gyro_x','gyro_y','gyro_z']
cols_to_use = ['accel_x','accel_y','accel_z','linaccel_x','linaccel_y','linaccel_z','gyro_x','gyro_y','gyro_z','labels']

Xy = di[cols_to_use]

train_rf = pd.DataFrame(columns = ['mean_x','median_x','stddev_x','mean_y','median_y','stddev_y','mean_z','median_z','stddev_z','labels'])
for i in range(0,len(Xy),50):
    v = Xy.iloc[i:(i + 50)].values
    y = v[:,9]
    X = pd.DataFrame(np.delete(v, 9, axis = 1))
    
    mean_acc = X[[0,1,2]].apply(lambda x: np.mean(x))
    median_acc = X[[0,1,2]].apply(lambda x: stats.median_abs_deviation(x))
    stddev_acc = X[[0,1,2]].apply(lambda x: np.std(x))
    
    mean_linacc = X[[3,4,5]].apply(lambda x: np.mean(x))
    median_linacc = X[[3,4,5]].apply(lambda x: stats.median_abs_deviation(x))
    stddev_linacc = X[[3,4,5]].apply(lambda x: np.std(x))
    
    mean_gyro = X[[6,7,8]].apply(lambda x: np.mean(x))
    median_gyro = X[[6,7,8]].apply(lambda x: stats.median_abs_deviation(x))
    stddev_gyro = X[[6,7,8]].apply(lambda x: np.std(x))
    
    row = {'mean_x_acc':mean_acc[0],'median_x_acc':median_acc[0],'stddev_x_acc':stddev_acc[0],
           'mean_y_acc':mean_acc[1],'median_y_acc':median_acc[1],'stddev_y_acc':stddev_acc[1],
           'mean_z_acc':mean_acc[2],'median_z_acc':median_acc[2],'stddev_z_acc':stddev_acc[2],
           
           'mean_x_linacc':mean_linacc[0],'median_x_linacc':median_linacc[0],'stddev_x_linacc':stddev_linacc[0],
           'mean_y_linacc':mean_linacc[1],'median_y_linacc':median_linacc[1],'stddev_y_linacc':stddev_linacc[1],
           'mean_z_linacc':mean_linacc[2],'median_z_linacc':median_linacc[2],'stddev_z_linacc':stddev_linacc[2],
           
           'mean_x_gyro':mean_gyro[0],'median_x_gyro':median_gyro[0],'stddev_x_gyro':stddev_gyro[0],
           'mean_y_gyro':mean_gyro[1],'median_y_gyro':median_gyro[1],'stddev_y_gyro':stddev_gyro[1],
           'mean_z_gyro':mean_gyro[2],'median_z_gyro':median_gyro[2],'stddev_z_gyro':stddev_gyro[2],
           
           'labels':stats.mode(y)[0][0]}
    
    train_rf = train_rf.append(row, ignore_index = True)

train_rf.to_csv('rf_trainingset_combined.csv', index = True)
    
    
