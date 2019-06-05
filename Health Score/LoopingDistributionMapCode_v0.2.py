import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import feather
import numpy as np
import os
import glob
import datetime
import configparser
import xgboost as xgb
import pickle
from shutil import copyfile
from itertools import accumulate
from functools import reduce
from datetime import datetime
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config_path =  'E:/CodesXGBoost_R/redistribution_config.ini' 
config.read(config_path)

### Parameters and Path Initialization ###
print("Reading Config File Inputs")
input_path = config['PATH']['input_path']
output_path = config['PATH']['output_path']
target =  config['PARAMETERS']['target']

if not os.path.exists(output_path):
    os.makedirs(output_path)

csv_files = glob.glob(input_path+ target +"*OuptFileResidualSum.csv")
#csv_files = [f for f in csv_files if 'OutputFile']


inp = {}
outp = {}


inp_min =  config['PARAMETERS']['inp_min'].split(",")
inp['min'] = [float(i) for i in inp_min] 
inp['max'] = [float(config['PARAMETERS']['inp_max']), inp['min'][0],inp['min'][1]]

outp_min = config['PARAMETERS']['outp_min'].split(",")
outp['min'] = [float(i) for i in outp_min] 
outp['max'] = [outp['min'][1],outp['min'][2],float(config['PARAMETERS']['outp_max'])]
outp['col_name'] = "disp"
outp['cat'] = config['PARAMETERS']['outp_cat'].split(",")

cols_reqd = config['PARAMETERS']['cols'].split(",")

date = datetime.datetime.now()
day = str(date.day)
month =  str(date.month)
year =  str(date.year)

exec_date = day+month+year


for file in csv_files:
    
     
    dat =pd.DataFrame()
    #dat['scr'] = (np.arange(1,61,1)-4)*200
    
    print("Reading Data")
    dat = pd.read_csv(file) 
    #### While testing
    
    ##### distribution remap  ######
    
    
    dat =  dat[cols_reqd]
    dat.columns = ['Machine','Date','Residual_Sum']
    mac =  list(set(dat.Device))[0]
    print(mac)
    
    inp['range'] = []
    outp['range'] = []
    
    for i in range(len(inp['min'])):
        inp['range'].append(abs(inp['min'][i]-inp['max'][i]))
        outp['range'].append(abs(outp['min'][i]-outp['max'][i]))
        
    dat['bin_range'] = np.where(dat['Residual_Sum'] < inp['min'][1],3,np.where(dat['Residual_Sum'] < inp['min'][0],2,1))
    dat['risk_catg'] = np.where(dat['bin_range']==1,outp['cat'][0],np.where(dat['bin_range']==2,outp['cat'][1],outp['cat'][2]))
    
    dat['perc_in_bin'] = float(config['PARAMETERS']['perc_in_bin'])
    dat['disp']    = float(config['PARAMETERS']['disp'])
    
    dat1 = dat[dat['bin_range'] == 1]
    dat2 = dat[dat['bin_range'] == 2]
    dat3 = dat[dat['bin_range'] == 3]
    
    
    dat1['perc_in_bin'] = np.round((np.minimum(dat1['Residual_Sum'], inp['max'][0]) - inp['min'][0])/(inp['range'][0] + 0.000001),4)
    dat2['perc_in_bin'] = np.round((dat2['Residual_Sum'] - inp['min'][1])/(inp['range'][1] + 0.000001),4)
    dat3['perc_in_bin'] = np.round((np.minimum(dat3['Residual_Sum'], inp['max'][2]) - inp['min'][2])/(inp['range'][2] + 0.000001),4)
    
    
    dat1['disp'] = np.round(outp['min'][0] + outp['range'][0] - dat1['perc_in_bin']*outp['range'][0],4)
    dat2['disp'] = np.round(outp['min'][1] + outp['range'][1] - dat2['perc_in_bin']*outp['range'][1],4)
    dat3['disp'] = np.round(outp['min'][2] + outp['range'][2] - dat3['perc_in_bin']*outp['range'][2],4)
        
      
    dat1['disp'] = np.where(dat1['Residual_Sum']>inp['max'][0],outp['min'][0],dat1['disp'])
    dat3['disp'] = np.where(dat3['Residual_Sum']<inp['min'][2],outp['max'][2],dat3['disp'])
    
    
    dat = pd.concat([dat1,dat2,dat3], axis = 0, ignore_index = True)
    dat.sort_values(['Device','TimeStamp'], ascending = [True,True], inplace = True)
     
    dat.drop('bin_range', axis = 1, inplace = True)   
    print("Write Final File")
    dat.to_csv(output_path +mac+ "_"+target+ "_OuptFileRedistributionMap.csv", index = False)    
    
copyfile(config_path, output_path +  target + exec_date + "_redistribution_config.ini")    
    
    