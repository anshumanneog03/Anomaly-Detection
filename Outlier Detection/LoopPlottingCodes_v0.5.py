###### Outlier Detection New ######

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
from itertools import accumulate
from functools import reduce

import matplotlib.pyplot as plt
from shutil import copyfile

config = configparser.ConfigParser()
config_path = 'E:/CodesXGBoost_R/config_outlier.ini'  
config.read(config_path)

machine_list = config['OTHER']['read_machine_list'].split(",")

input_path = config['PATH']['input_directory']
featherfile = config['PATH']['featherfile']
filename = config['PATH']['filename']
err_var = config['OTHER']['err_var']
target  =  config['OTHER']['target']
holdout_ref_date = config['OTHER']['holdout_ref_date']
output_path =  config['PATH']['output_directory']
first_run = config['OTHER']['first_run']
targeterr_mean =  config['OTHER']['err_mean']
targeterr_sd =  config['OTHER']['err_sd']

path_v =  config['PATH']['model_directory']
validation_filename = config['PATH']['validation_filename'] 

cols_req = config['OTHER']['columns_req'].split(',')
cols_req.append(target)
cols_req.append(err_var)

if not os.path.exists(output_path):
    os.makedirs(output_path)


model_day =  config['OTHER']['model_day']
model_month =  config['OTHER']['model_month']
model_year =  config['OTHER']['model_year']

model_date =  model_year + "_" +  model_month + "_" + model_day + "_"




date = datetime.datetime.now()
day = str(date.day)
month =  str(date.month)
year =  str(date.year)

exec_date = day+month+year

if first_run == 'FALSE':
    val_df = pd.read_csv(path_v + model_date + target + "_"+validation_filename + ".csv")


for k in range(len(machine_list)):
    
    
    
    if first_run == 'TRUE':
        
        if(featherfile == 'TRUE'):
            dt1 = feather.read_dataframe(input_path + machine_list[k]+target+filename+'.feather')
        else:
            dt1 = pd.read_csv(input_path + machine_list[k]+target+filename+'.csv')
    else:
        dt1 = pd.read_csv(input_path + target +  machine_list[k] + "_" + model_date +  filename + ".csv") 
        
        
        
    dt1 =  dt1[cols_req]
    print(set(dt1.Device))
    
    
    
    if first_run == 'TRUE':
    
        target_mean = np.mean(dt1[err_var][dt1['train_val'] == 'v'])
        target_sd = np.std(dt1[err_var][dt1['train_val'] == 'v'])
    else:
        target_mean = val_df['VALIDATION_MEAN'][val_df['MACHINE'] == machine_list[k]].tolist()[0]     ###### This code as of now as of now will take a single input values for all machines, it will be chnaged later to take machine wise std and mean
        target_sd = val_df['VALIDATION_STD'][val_df['MACHINE'] == machine_list[k]].tolist()[0]
    
    dt1[target+'_err_z'] = np.abs((dt1[err_var]- target_mean))/target_sd   #### This can be changed to calculate absolute values using np.abs()
    
    
    
    z_risk_threshold = float(config['OTHER']['z_risk_threshold'])  
    z_risk_drain_threshold = float(config['OTHER']['z_risk_drain_threshold']) 
    
    dt1.sort_values(['Device','TimeStamp'], ascending  = [True,True], inplace = True)
    
    
    
    if first_run == 'TRUE':
        dt_h = dt1[dt1['TimeStamp'].apply(lambda x: x[0:7])>= holdout_ref_date]
    else:
        dt_h = dt1
    
    dt_h[target+'_z_risk_flag'] = np.where(dt_h[target+'_err_z']> z_risk_threshold,1,0)
    dt_h[target+'_z_risk_accum'] =  np.where(dt_h[target+'_err_z'] > z_risk_threshold,dt_h[target+'_err_z'] - z_risk_threshold,0)
    dt_h[target+'_z_risk_drain'] = np.where((dt_h[target+'_err_z'] < z_risk_drain_threshold) & (dt_h[target+'_err_z'] > 0),dt_h[target+'_err_z'] - z_risk_drain_threshold,0)
    dt_h[target+'_z_risk_net']  = dt_h[target+'_z_risk_accum'] + dt_h[target+'_z_risk_drain']
    
    
    
    dt_h_riskscore = dt_h.copy()
    
    #### Converting into Datetime ####
    
    dt_h_riskscore['TimeStamp'] = pd.to_datetime(dt_h_riskscore['TimeStamp'])
    
    ### Extracting Dates only ###
    
    dt_h_riskscore['Date'] = [dt.date() for dt in dt_h_riskscore['TimeStamp']]
    
    #### Group By Sum per day
    
    daily_df = dt_h_riskscore.groupby(['Device','Date'])[target + '_z_risk_net'].agg(['sum']).reset_index()
    
    daily_df.columns = ['Machine','Date','Incremental_Daily_Residual']
    
    #daily_df.to_csv(output_path + target +exec_date+"_"+machine_list[k]+"_MachineDate_SumofNetError.csv", index =  False)
    
    
    
    
    
    def cumsum_bounded(z, upper_bound = 500000, lower_bound = 0):
        return list(accumulate(z, lambda x,y:min(upper_bound,max(lower_bound,x+y))))
    
    cat_df = dt_h_riskscore.groupby('Device')[target + '_z_risk_net'].agg([cumsum_bounded])
    
    cat_df['Device'] = cat_df.index
    
    cat_list = cat_df['Device'].tolist()
    
    main_df = pd.DataFrame()
    for i,j in enumerate(cat_list):
        print(i)
        df = dt_h_riskscore[dt_h_riskscore['Device'] == j]
        df['Residual_Sum'] = cat_df['cumsum_bounded'][i]
        main_df = pd.concat([main_df,df],axis =0)
        
    main_df.sort_index(axis = 0, inplace = True)
    
    
    
    
    
    dt_h_riskscore = main_df
    
    dt1_riskscore = dt_h_riskscore
    
    
    dt1_riskscore.to_csv(output_path +  target +exec_date+"_"+machine_list[k]+"_OuptFileResidualSum.csv",index = False)
    
    ################# ADDITONAL OUTPUTS ##############################

    ## 1. DAILY RESDIUAL SUM 
    ## 2. CUMSUM BOUNDED ON DAILY SUM OF Z RISK NET
    
    ### DAILY RESIDUAL SUM ###
    
    dailyResidualSum =  dt1_riskscore.groupby(['Device','Date'])['Residual_Sum'].agg(['sum'])
    #dailyResidualSum.to_csv(output_path + target + exec_date +"_"+machine_list[k]+"_dailyResidualSum.csv")
    
    
    ### CUMSUM BOUNDED ON DAILY SUM OF Z RISK NET
    
    dailySumZRiskCumsum = daily_df.groupby('Machine')['Incremental_Daily_Residual'].agg([cumsum_bounded])
    dailySumZRiskCumsum['Machine'] =  dailySumZRiskCumsum.index
    mac_list_2 = dailySumZRiskCumsum['Machine'].tolist()
    
    main_df_1 = pd.DataFrame()
    for i,j in enumerate(mac_list_2):
        print(i)
        print(j)
        df = daily_df[daily_df['Machine'] == j]
        df['Z_RISK_NET_CMSM'] = dailySumZRiskCumsum['cumsum_bounded'][i]
        main_df_1 = pd.concat([main_df_1,df],axis =0)
        
    main_df_1.sort_index(axis = 0, inplace = True)
    
    #main_df_1.to_csv(output_path + target + exec_date+ "_"+machine_list[k]+"_dailySumZRiskCumsum_ABS.csv")
    
    
    
    
    
    
    
    
    ##### Plotting #####
    
    plotting =  config['OTHER']['plotting']
    if plotting == 'TRUE':
        n = dt1_riskscore.shape[0]
        
        dt1_riskscore_plot  =  dt1_riskscore.sample(round(n/30))
        dt1_riskscore_plot['TimeStamp'] = pd.to_datetime(dt1_riskscore_plot['TimeStamp'])
        
        
        
        device_list = list(set(dt1['Device']))
        dev_list_read =  config['OTHER']['device_list'].split(",")
        
        ## Additional Loop Code
        dev_list_read = []
        dev_list_read.append(machine_list[k])
        
        
        
        
        dev_to_read = []
        for i in device_list:
            for j in dev_list_read:
                if j in i:
                    dev_to_read.append(i)
        
        data = pd.DataFrame()
        for i in range(len(dev_to_read)):
            dt = dt1_riskscore_plot[dt1_riskscore_plot['Device'] == dev_to_read[i]]
            data =  pd.concat([data,dt], axis = 0, ignore_index = True)
            
        data.sort_values(['Device','TimeStamp'], ascending = [True, True], inplace = True)
        
        
        ax = data.pivot(index='TimeStamp', columns='Device', values='Residual_Sum').plot()
        
        fig = ax.get_figure()
        fig.savefig(output_path+target+"_"+machine_list[k]+'_ResidualSum.png')
    

copyfile(config_path, output_path + target + "_"+ exec_date + "_config_outlier_detect.ini" )   
    
    
    
