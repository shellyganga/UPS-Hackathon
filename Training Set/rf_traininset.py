import pandas as pd
import numpy as np
from scipy import stats

ddir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'
di = pd.read_csv(ddir + 'trainingset_labeled.csv')

di = di[di['labels'] != 7]

Xy = di[['x', 'y', 'z','labels']]

train_rf = pd.DataFrame(columns = ['mean_x','median_x','stddev_x','mean_y','median_y','stddev_y','mean_z','median_z','stddev_z','labels'])
for i in range(0,len(Xy),50):
    v = Xy.iloc[i:(i + 50)].values
    y = v[:,3]
    X = pd.DataFrame(np.delete(v, 3, axis = 1))
    
    mean = X.apply(lambda x: np.mean(x))
    median = X.apply(lambda x: stats.median_abs_deviation(x))
    stddev = X.apply(lambda x: np.std(x))
    
    row = {'mean_x':mean[0],'median_x':median[0],'stddev_x':stddev[0],
           'mean_y':mean[1],'median_y':median[1],'stddev_y':stddev[1],
           'mean_z':mean[2],'median_z':median[2],'stddev_z':stddev[2],'labels':stats.mode(y)[0][0]}
    
    train_rf = train_rf.append(row, ignore_index = True)

train_rf.to_csv('rf_trainingset.csv', index = True)
    
    
