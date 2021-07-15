import pandas as pd
import numpy as np


def demo_data(X, time_steps, step, model):
    XX = X[['x','y','z']]
        
    Xs = [], []
    for i in range(0, len(XX) - time_steps, step):
        v = XX.iloc[i:(i + time_steps)].values
        Xs.append(v)
        
    # model predict with Xs
    
    t_inseconds = time_steps * (1/50)
    
    
    
    return x_out



# Create Dataset
TIME_STEPS = 10
STEP = 5

X_train  = create_dataset(
    di[['x', 'y', 'z']],
    TIME_STEPS,
    STEP
)

