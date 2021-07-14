import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

## non aggressive (0), aggressive right turn (1), aggressive left turn (2), 
#aggressive lane change right (3), aggressive lane change left (4), aggressive acceleration (5), aggressive stop (6)
event_dict = {'evento_nao_agressivo':0, 'curva_direita_agressiva':1,'curva_esquerda_agressiva':2,
              'troca_faixa_direita_agressiva':3, 'troca_faixa_esquerda_agressiva':4,
              'aceleracao_agressiva':5,'freada_agressiva':6}

ddir = 'C:/users/spwiz/Downloads/driverBehaviorDataset-master/driverBehaviorDataset-master/data/'
odir = 'C:/users/spwiz/Documents/GitHub/UPS-Hackathon-Resources/Training Set/'

di16 = pd.read_csv(ddir + '16/acelerometro_terra.csv')
di17 = pd.read_csv(ddir + '17/acelerometro_terra.csv')
di20 = pd.read_csv(ddir + '20/acelerometro_terra.csv')
di21 = pd.read_csv(ddir + '21/acelerometro_terra.csv')

di16_gt = pd.read_csv(ddir + '16/groundTruth.csv')
di17_gt = pd.read_csv(ddir + '17/groundTruth.csv')
di20_gt = pd.read_csv(ddir + '20/groundTruth.csv')
di21_gt = pd.read_csv(ddir + '21/groundTruth.csv')

dis = [di16,di17,di20,di21]
dis_label = [di16_gt, di17_gt, di20_gt, di21_gt]
for i in range(4):
    di = dis[i]
    dil = dis_label[i]
    
    dil['evento'] = dil['evento'].replace(event_dict)
    
    print(pd.unique(dil['evento']))
    
    print(di.columns)   
    print(di.shape[0])
    
    di['timestamp']= pd.to_datetime(di['timestamp'])
    timeframe = di['timestamp'].max() - di['timestamp'].min()
    print(timeframe)
    di.index = pd.DatetimeIndex(di.timestamp)
    
    di['uptimenanos_diff'] = di['uptimeNanos'] - np.full((di.shape[0],),di['uptimeNanos'].min())
    di['uptimesecond_diff'] = round(di['uptimenanos_diff']*1e-9,1)
    
    times = np.arange(di['uptimesecond_diff'].min(), di['uptimesecond_diff'].max(), 0.1)
    
    #df = di.groupby(di['uptimesecond_diff'])[['x','y','z']].mean().reindex()
    
    #series = pd.Series(di['x'], index = di.index)
    #series.index = pd.to_datetime(series.index)
    #mean = series.resample('1S').mean()
    
    i = 0
    dilabel = pd.DataFrame()
    for index, row in dil.iterrows():
        start, end = row[' inicio'], row[' fim']
        
        colname = 'col' + str(i)
        dilabel[colname] = np.where((di['uptimesecond_diff'] >= start) & (di['uptimesecond_diff'] <= end), row[0], np.nan)
        
        print(pd.unique(dilabel[colname]))
        
        i += 1
    
    
    print(dilabel.shape[1] == dil.shape[0])
    totlabels = dilabel.loc[:,list(dilabel.columns)].sum(axis=1, min_count=1)
    totlabels.name = 'labels'
    print(pd.unique(totlabels))
    
    di['labels'] = list(totlabels)
    print(pd.unique(di['labels']))
        
df = pd.concat(dis)

df = df.drop(columns = ['uptimenanos_diff','uptimesecond_diff','timestamp'])

df['uptimenanos_diff'] = df['uptimeNanos'] - np.full((df.shape[0],),df['uptimeNanos'].min())
df['uptimesecond_diff'] = round(df['uptimenanos_diff']*1e-9,1)
df = df.sort_values(by = 'uptimenanos_diff', ascending = True)

#x = pd.unique(di['uptimesecond_diff'])
print(pd.unique(df['labels']))

df = df.replace([-111.0],np.nan)

df.to_csv(odir + 'trainingset_labeled.csv', index = True, na_rep = np.nan)
