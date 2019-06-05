####### XG BOOST Code Conversion ########



#### Feather has been installed in the system #####



#### Library Imports ####
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
from sklearn.utils import check_array
from sklearn.metrics import r2_score
import time
import gc
from shutil import copyfile


date_now =  datetime.datetime.now()
yr = str(date_now.year)
mnth =  str(date_now.month)
dy = str(date_now.day)


proc_st_time = time.time()
##### The path for the config file has been hardcoded ######
config = configparser.ConfigParser()
config_path =  'G:/CodesXGBoost_R/config.ini'
config.read(config_path)

print("\n****** STARTING PROCESS ******* \n")


### Param_io functions when invoked creates some support directories using the paths it reads  from the config file ###

def define_param_io():
    
    
    param_io = {} # Initializing the dictionary to be populated below
    
    param_io['home_directory']     =  config['PATH']['home_dir']       ### Directory has to be created first and code and config files can be put here 
    param_io['input_directory']    =  config['PATH']['data_dir']       ### Directory has to be created first as it expects the input data to sit here
    param_io['test_directory']     =  config['PATH']['test_data_dir']  ### Directory has to be created first as test data is expected to sit here
    
    
    param_io['code_directory']     =  param_io['home_directory']  +"R_data_mining/" ### Not Needed
    param_io['working_directory']  =  param_io['home_directory']+"main/"     
    
    if not os.path.exists(param_io['working_directory']):
        os.makedirs(param_io['working_directory'])
    
    param_io['model_directory']    =  param_io['home_directory'] + "models/" 
    if not os.path.exists(param_io['model_directory'] ):
        os.makedirs(param_io['model_directory'] )    
    
    
    os.chdir(param_io['input_directory'])
    
    
    file_pattern = config['DATA NAME']['data_pattern']
    file_loc = glob.glob(param_io['input_directory']+file_pattern+"*.feather")
    file_no = len(file_loc)
                
            
    param_io['file_no'] = file_no
    param_io['file_loc'] = file_loc
        

    
    param_io['dbc_directory']    =     param_io['home_directory'] + "dbc/"
    if not os.path.exists(param_io['dbc_directory'] ):
        os.makedirs(param_io['dbc_directory'] )    
    
    
    param_io['output_directory']    =  param_io['home_directory'] + "data_output/"
    if not os.path.exists(param_io['output_directory']):
        os.makedirs(param_io['output_directory'])    
    
    
    
    param_io['skip_feather']            =  config['FEATHER']['skip']
    
    return param_io

### Param Prep contains the target variable information and the columns to be read when the data  is read in.
    


def define_param_prep():
    
    
    
    param_prep = {} # Initializing the dictionary to be populated below
    
    param_prep['cols_read'] = config['COLUMNS']['cols_read'].split(",")
    param_prep['global_mean'] = float(config['TARGET']['target_global_mean']) # this global mean is not being used in the code as the code calculates mean and stddev itself
    param_prep['target'] = config['TARGET']['target_var']

    return param_prep


##### define_param_model contains  the xg boost parameters and other support parameters that it derives from the config file and is used in the code further
    

def define_param_model(use_full_data):
    
    
    
     
    param_model = {}
    param_model['max_depth'] = int(config['PARAM MODEL']['max_depth'])
    param_model['eta'] = int(config['PARAM MODEL']['eta'])
    param_model['objective'] = config['PARAM MODEL']['objective']
    param_model['num_round'] = int(config['PARAM MODEL']['num_round'])
    
    param_model['rand_seed'] = int(config['PARAM MODEL']['seed'])
    param_model['train'] = config['PARAM MODEL']['train_flag'] 
    param_model['score'] = config['PARAM MODEL']['score']
    param_model['holdout_ref_date'] = config['PARAM MODEL']['hold_out_date']
    param_model['target'] =  config['TARGET']['target_var']
    
    if use_full_data == 'TRUE':
        param_model['train_perc'] = float(config['PARAM MODEL']['train_perc_T'])
        param_model['valid_perc'] = float(config['PARAM MODEL']['valid_perc_T'])
    else:
        param_model['train_perc'] = float(config['PARAM MODEL']['train_perc_F'])
        param_model['valid_perc'] = float(config['PARAM MODEL']['valid_perc_F'])
        
    return param_model

#### param_dbc contains dbc dictionary which contains the a dataframe with cat variables , inter size and correpsonding expression to be invoked while creating that  particular dbc feature

def define_param_dbc():
    
    
    
    dbc = {}
    
    var_name = config['DBC']['var_names'].split(",")
    abbrev  = config['DBC']['abbrev'].split(",")

    dbc['abbrev'] = pd.DataFrame({'var_name':var_name,'abbrev':abbrev})
    dbc['target'] = config['TARGET']['target_var']
    dbc['target_global_mean'] = float(config['TARGET']['target_global_mean'])      # this initial value is not used
    dbc['target_global_stdev'] = float(config['TARGET']['target_global_stdev'])    # this initial value is not used
    
    dbc['var_catg'] = "cat_"
    dbc['var_prefix'] = "dbc_"
    dbc['save_table_prefix'] = param_io['dbc_directory'] + "dbc_tabl_"
    
    dbc['min_cell_cnt'] = config['DBC']['min_cell_cnt']
    dbc['contin_to_n_bin'] = config['DBC']['contin_to_n_bin']
    dbc['train'] = config['DBC']['train']
    dbc['score'] = config['DBC']['score']
    dbc['rank'] = config['DBC']['rank']
    dbc['rank_thresh'] = config['DBC']['rank_thresh']
    dbc['delete_tmp_vars'] = config['DBC']['delete_tmp_vars']
    dbc['eval_top_n_perc'] =  config['DBC']['eval_top_n_perc']
    dbc['dlm'] = "|"
    dbc['type'] = "freq"
    dbc['rank_func'] = "mean"
    dbc['measure_mean'] = config['DBC']['measure_mean']
    dbc['measure_stdev'] = config['DBC']['measure_stdev']
    dbc['measure_cnt'] = int(config['DBC']['measure_cnt'])
    dbc['measure_bayes'] = config['DBC']['measure_bayes']
    dbc['measure_kappa'] = config['DBC']['measure_kappa']
    
    
    
    ivar = ["ambt","dvic","powa","rots","tdif","winv","ambt_dvic",
      "ambt_powa",
      "ambt_rots",
      "ambt_tdif",
      "powa_rots",
      "powa_tdif",
      "ambt_powa_rots",
      "ambt_powa_tdif"]

   
    dt_expression_a = ["dt['cat_ambt'] = dt['T_AMB_Avg_bin'].apply(lambda x: str(x))","dt['cat_dvic'] = dt['Device'].apply(lambda x: x[-2:])",
                      "dt['cat_powa'] = dt['P_ACT_Avg_bin'].apply(lambda x: str(x))",
                      "dt['cat_rots'] = dt['N_ROT_PLC_Avg_bin'].apply(lambda x: str(x))",
                      "dt['cat_tdif'] = dt['oper_temp_rise_bin'].apply(lambda x: str(x))",
                      "dt['cat_winv'] = dt['V_WIN_Avg_bin'].apply(lambda x: str(x))",
                      "dt['cat_ambt_dvic'] = dt[['cat_ambt','cat_dvic']].apply(lambda x: '|'.join(x),axis =1)",
                      "dt['cat_ambt_powa'] = dt[['cat_ambt','cat_powa']].apply(lambda x: '|'.join(x),axis = 1)",
                      "dt['cat_ambt_rots'] = dt[['cat_ambt','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)",
                      "dt['cat_ambt_tdif'] = dt[['cat_ambt','cat_tdif']].apply(lambda x: '|'.join(x), axis = 1)",
                      "dt['cat_powa_rots'] = dt[['cat_powa','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)",                      
                      "dt['cat_powa_tdif'] = dt[['cat_powa','cat_tdif']].apply(lambda x: '|'.join(x),axis = 1)",
                      "dt['cat_ambt_powa_rots'] = dt[['cat_ambt_powa','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)",
                      "dt['cat_ambt_powa_tdif'] = dt[['cat_ambt_powa','cat_tdif']].apply(lambda x: '|'.join(x), axis  = 1)"
                      
                      ] #this entire set of expression was meant to be used dynamically in a loop in the prep_dbc code, instead it has been hardcoded in there
    
    inter_size = [1,1,1,1,1,1,2,2,2,2,2,2,3,3]
    
    inter_substitution = [
      "", "", "", "", "", "",
      "dbc_mean_ambt,dbc_mean_dvic,dbc_mean_ambt_dvic", 
      "dbc_mean_ambt,dbc_mean_powa,dbc_mean_ambt_powa", 
      "dbc_mean_ambt,dbc_mean_rots,dbc_mean_ambt_rots", 
      "dbc_mean_ambt,dbc_mean_tdif,dbc_mean_ambt_tdif", 
      "dbc_mean_powa,dbc_mean_rots,dbc_mean_powa_rots", 
      "dbc_mean_powa,dbc_mean_tdif,dbc_mean_powa_tdif", 
      "dbc_mean_ambt_powa,dbc_mean_ambt_rots,dbc_mean_powa_rots,dbc_mean_ambt_powa_rots",
      "dbc_mean_ambt_powa,dbc_mean_ambt_tdif,dbc_mean_powa_tdif,dbc_mean_ambt_powa_tdif"
    ]
    
    dbc['dvar'] = pd.DataFrame({'ivar': ivar,'dt_expression_a':dt_expression_a,'inter_size': inter_size,'inter_substitution':inter_substitution })
    
    return dbc



### ee_abnorm_filter takes a single data set,removes the the null, erroneous machines, selectively removes the machines
### with their anomaly periods and also records with non-operational values 

def ee_filter_abnorm(dt, param_prep):
    
    dtn = dt[param_prep['cols_read']]  # repetition of  step from the data ingestion step , to double check
    
    
    filter_no = int(config['FILTER ABNORM']['filter_no'])
    filters = config['FILTER ABNORM']['filters'].split(",")
    filter_values = config['FILTER ABNORM']['filter_values'].split(",")
    filter_values = [float(i) for i in filter_values]
    for i in range(filter_no):
        dtn = dtn[dtn[filters[i]] > filter_values[i]]
        
    anomalous_machines = config['FILTER ABNORM']['anomaly_mac'].split(",")
    
        
    if anomalous_machines[0] != '':
        dtn['yr_mon_day'] = dtn['TimeStamp'].str[:10]
        
        start_dates = config['FILTER ABNORM']['start_date'].split(",")
        end_dates   = config['FILTER ABNORM']['end_date'].split(",")
        mac_no = len(anomalous_machines)
        
        for i in range(mac_no):
            
            dtn['row_to_drop'] = 0
            print("Removing Machine",i+1)
            dtn['row_to_drop'] =  np.where((dtn['Device'] ==  anomalous_machines[i]) & (dtn['yr_mon_day']>= start_dates[i]) & (dtn['yr_mon_day'] <= end_dates[i]),1,0)
            dtn = dtn[dtn['row_to_drop'] != 1]
            dtn.drop('row_to_drop', axis = 1, inplace =True)
    
        dtn.drop('yr_mon_day',axis = 1, inplace = True)
    
    remove_machine = config['FILTER ABNORM']['remove_machine'].split(",")
     
    if len(remove_machine) > 1:
        for i in range(len(remove_machine)):
            dtn = dtn[dtn['Device']!=remove_machine[i]]
    elif len(remove_machine) == 1 and remove_machine[0] != '':
        dtn = dtn[dtn['Device']!=remove_machine[0]]
    
    print(list(set(dtn.Device)))   
    
    
    dtn.dropna(axis = 0 , inplace = True)
    
    return dtn
####### PREP DBC WITHOUT EXEC #######
    
### The prep_dbc function does the task of creating dbc features. It first creates the categorical bin variables 
### based upon which it gets the bin wise output feature's mean and std. 

### The Training data generates this dbc look up files which are also used for validation and holdout data to generate the 
### dbc feature.
    


date_now =  datetime.datetime.now()
yr = str(date_now.year)
mnth =  str(date_now.month)
dy = str(date_now.day)


def prep_dbc(dt, dbc, param_io, param_prep,mac,sep_model):
    
    mac = mac
    if 'log_level' in globals():
        if log_level >= 2:
            print("\n ------------------------------------------\n")
            print(". prep_dbc( dt, dbc, param_io, param_prep) \n")
            print(". starting: ", str(datetime.datetime.now()), "\n")
            print("\n ------------------------------------------\n\n")
    
    dt['cat_ambt'] = dt['T_AMB_Avg_bin'].apply(lambda x: str(x))
    print('now creating '+'var 1')
    dt['cat_dvic'] = dt['Device'].apply(lambda x: x[-2:])
    print('now creating '+'var 2')
    dt['cat_powa'] = dt['P_ACT_Avg_bin'].apply(lambda x: str(x))
    print('now creating '+'var 3')
    dt['cat_rots'] = dt['N_ROT_PLC_Avg_bin'].apply(lambda x: str(x))
    print('now creating '+'var 4')
    dt['cat_tdif'] = dt['oper_temp_rise_bin'].apply(lambda x: str(x))
    print('now creating '+'var 5')
    dt['cat_winv'] = dt['V_WIN_Avg_bin'].apply(lambda x: str(x))
    print('now creating '+'var 6')
    dt['cat_ambt_dvic'] = dt[['cat_ambt','cat_dvic']].apply(lambda x: '|'.join(x),axis =1)
    print('now creating '+'var 7')
    dt['cat_ambt_powa'] = dt[['cat_ambt','cat_powa']].apply(lambda x: '|'.join(x),axis = 1)
    print('now creating '+'var 8')
    dt['cat_ambt_rots'] = dt[['cat_ambt','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)
    print('now creating '+'var 9')
    dt['cat_ambt_tdif'] = dt[['cat_ambt','cat_tdif']].apply(lambda x: '|'.join(x), axis = 1)
    print('now creating '+'var 10')
    dt['cat_powa_rots'] = dt[['cat_powa','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)
    print('now creating '+'var 11')
    dt['cat_powa_tdif'] = dt[['cat_powa','cat_tdif']].apply(lambda x: '|'.join(x),axis = 1)
    print('now creating '+'var 12')
    dt['cat_ambt_powa_rots'] = dt[['cat_ambt_powa','cat_rots']].apply(lambda x: '|'.join(x), axis = 1)
    print('now creating '+'var 13')
    dt['cat_ambt_powa_tdif'] = dt[['cat_ambt_powa','cat_tdif']].apply(lambda x: '|'.join(x), axis  = 1)
    print('now creating '+'var 14')
    for d in range(dbc['dvar'].shape[0]):
        
        if 'log_level' in globals():
            if log_level >= 3:
                 print("d = ", d+1, ":  i_var = ", dbc['dvar']['ivar'][d], "\n")
        
        
        if dbc['train'] == 'TRUE':
            df = dt.groupby('cat_'+dbc['dvar']['ivar'][d])[dbc['target']].agg(['mean','count'])
            df['cat_'+dbc['dvar']['ivar'][d]] = df.index
            df.rename(index = str, columns = {'count':'dbc_cnt','mean':'dbc_mean_'+dbc['dvar']['ivar'][d]}, inplace = True)
            
            
                        
            
        
            if dbc['measure_stdev'] == 'TRUE':
                
                tmp = dt.groupby('cat_'+dbc['dvar']['ivar'][d])[dbc['target']].agg(['std'])
                tmp['cat_'+dbc['dvar']['ivar'][d]] = tmp.index
                tmp.rename(index = str, columns = {'std':'dbc_stdev_'+dbc['dvar']['ivar'][d]}, inplace = True)
                
            
                ## joining the tmp and tab_ tables
                
                df =  df.join(tmp.set_index('cat_'+dbc['dvar']['ivar'][d]), on = 'cat_'+dbc['dvar']['ivar'][d])
                
                
            
                
            df = df[df['dbc_cnt'] > int(dbc['min_cell_cnt']) ]   ##### Change this in the Config
            #df = df[df['dbc_cnt'] > int(20) ]
            
            
            if not os.path.exists(param_io['dbc_directory']+mac+"/"+param_prep['target'] + "_" + dy+"_"+mnth+"_"+yr  +"/" ):
                os.makedirs(param_io['dbc_directory']+mac+"/" +param_prep['target']+  "_" + dy+"_"+mnth+"_"+yr  +"/") 

                             
            df.to_csv(param_io['dbc_directory']+mac+"/"+param_prep['target']+ "_" + dy+"_"+mnth+"_"+yr  +"/"+'dbc_'+ dbc['dvar']['ivar'][d]+'.csv', index = False)
            
        if (dbc['score']== 'TRUE') & (dbc['train'] == 'FALSE'):
            
            
            if dbc['measure_cnt'] == 2:
                df = pd.read_csv(param_io['dbc_directory']+mac+"/"+param_prep['target']+    "_" + dy+"_"+mnth+"_"+yr     +"/"+'dbc_'+ dbc['dvar']['ivar'][d]+'.csv',dtype = {'dbc_mean_'+dbc['dvar']['ivar'][d] : np.float64,'dbc_cnt': np.int32,'cat_'+dbc['dvar']['ivar'][d]:object,'dbc_stdev_'+dbc['dvar']['ivar'][d]:np.float64})
                print("DBC File Read for Scoring")
                
        
        if dbc['score'] =='TRUE':
            if dbc['dvar']['inter_size'][d] == 1:
                
                dt =  dt.join(df.set_index('cat_'+dbc['dvar']['ivar'][d]), on = 'cat_'+dbc['dvar']['ivar'][d] )
                dt['dbc_mean_'+dbc['dvar']['ivar'][d]]= dt['dbc_mean_'+dbc['dvar']['ivar'][d]].fillna(dbc['target_global_mean']) 
                
               
                if dbc['measure_stdev'] == 'TRUE':
                   dt['dbc_stdev_'+dbc['dvar']['ivar'][d]] =     dt['dbc_stdev_'+dbc['dvar']['ivar'][d]].fillna(dbc['target_global_stdev'])
                   

                dt.drop('dbc_cnt', axis = 1, inplace = True)
            if dbc['dvar']['inter_size'][d] >= 2:
                dt =  dt.join(df.set_index('cat_'+dbc['dvar']['ivar'][d]), on = 'cat_'+dbc['dvar']['ivar'][d] )
                dt['dbc_mean_'+dbc['dvar']['ivar'][d]]= dt['dbc_mean_'+dbc['dvar']['ivar'][d]].fillna(-1)
                
                
                col_list= dbc['dvar']['inter_substitution'][d].split(",")
                M = dt[col_list]
                M['global_mean'] = dbc['target_global_mean']
                
                
                dt['dbc_mean_'+dbc['dvar']['ivar'][d]] = M.max(axis = 1)
                
                if dbc['measure_stdev'] == 'TRUE':
                    std_col_list = [col.replace('mean','stdev') for col in col_list]
                    M2 = dt[std_col_list]
                    M2['global_stdev']= dbc['target_global_stdev']
                    max_val = M.max(axis = 1 )
                    M_names = M.columns.tolist()
                    M2_names = M2.columns.tolist()
                    M_tmp = pd.concat([M,M2], axis  = 1)
                    M_tmp['max_val'] = max_val
                    M_tmp['selected_stdev'] = ""
                    
                    for c in range(len(M_names)):
                        mean_var = M_names[c]
                        stdev_var = M2_names[c]
                        M_tmp['selected_stdev'] = np.where(M_tmp['max_val']==M_tmp[mean_var],M_tmp[stdev_var],M_tmp['selected_stdev'])
            
                    dt['dbc_stdev_'+dbc['dvar']['ivar'][d]] = M_tmp['selected_stdev']
                    
                  
                dt.drop('dbc_cnt', axis  = 1, inplace = True)
              
     
    if dbc['delete_tmp_vars'] == 'TRUE':
        var_list = dt.columns.tolist()
        prefix1 = "cat_"
        pref_len1 = len(prefix1)
        prefix2 = "d_"
        pref_len2 = len(prefix2)
        
        for v in range(len(var_list)):
            if (var_list[v][0:pref_len1] == prefix1) or (var_list[v][0:pref_len2] == prefix2):
                dt.drop(var_list[v], axis = 1, inplace = True)
    
    


    cols  =  dt.columns.tolist()
    cols = [c for c in cols if 'dbc' in c]
    for k in range(len(cols)):
        dt[cols[k]].replace(-1, dbc['target_global_mean'])
        
     
    return dt  

#### ee_prep_d1 is basically responsible for preparing the data before the modelling. this includes feature engineering as in dbc 
### features and z transform variables. This code has been generalized to handle current,power and temperature output features
### and is also flexible in using dbc features.

def ee_prep_d1(dtn, param_io, param_prep, dbc, param_model, skip_tagging,i,sep_model):
    mac = i
    use_dbc=  config['PARAM MODEL']['use_dbc']
    dtn.sort_values(['Device','TimeStamp'], ascending  = [True,True], inplace = True)  
    sep_model =sep_model
    dep_var =  config['TARGET']['target_type']
    if dep_var == 'temp' and use_dbc == 'TRUE':
        dtn[param_prep['target']+'_prior'] = dtn[param_prep['target']].shift(1)
        dtn[param_prep['target']+'_prior'] = dtn[param_prep['target']+'_prior'].fillna(dbc['target_global_mean'])
        dtn['oper_temp_rise_avg'] = dtn[param_prep['target']+'_prior']- dtn['T_AMB_Avg']
        dtn['oper_temp_rise_avg'] = np.where(dtn['oper_temp_rise_avg'] < 0,0,dtn['oper_temp_rise_avg'])
    elif dep_var != 'temp'  and use_dbc == 'TRUE':
        dtn['oper_temp_rise_avg'] = dtn['BL1_ACT_Avg']
        
     
    if dep_var != 'temp':
        print("Creating Turbulence Intensity")
        dtn['TI'] = dtn['V_WIN_StdDev']/(dtn['V_WIN_Avg'] + 0.001)
    
    
    dtn.sort_values(['Device','TimeStamp'], ascending  = [True, True], inplace = True)
    
###### TO DO
##Need to change this to only create bins using train and validation   
##
    if use_dbc == 'TRUE':
        
        dtn_tv = dtn[dtn['train_val'] != 'h']
        dtn_h =  dtn[dtn['train_val'] == 'h']
        
        var =  config['DBC']['var_names'].split(",")
        
        dtn_tv[var[0]+'_bin'],bin_var  = pd.qcut(dtn_tv[var[0]],q = int(dbc['contin_to_n_bin']), labels = [1,2,3,4,5,6,7,8,9,10], retbins = True)
        
        #bin_var = pd.qcut(dtn['T_AMB_Avg_bin'],q = int(dbc['contin_to_n_bin']),retbins = True)
        #bin_list =  bin_var[1][1:].tolist()
        #bin_df = pd.DataFrame({'T_AMB_Avg_bin':bin_list})
        
        bin_df = pd.DataFrame({var[0] + '_bin':bin_var[1:]})
        
        
        dtn_tv['oper_temp_rise_bin'],  bin_var = pd.qcut(dtn_tv['oper_temp_rise_avg'],q = int(dbc['contin_to_n_bin']),labels = [1,2,3,4,5,6,7,8,9,10],  retbins = True)
        
        #bin_var = pd.qcut(dtn['oper_temp_rise_bin'],q = int(dbc['contin_to_n_bin']),retbins = True)
        #bin_list =  bin_var[1][1:].tolist()
        bin_df['oper_temp_rise_bin'] = bin_var[1:]
        
        
        
        dtn_tv[var[2]+'_bin'], bin_var = pd.qcut(dtn_tv[var[2]],q = int(dbc['contin_to_n_bin']), labels = [1,2,3,4,5,6,7,8,9,10], retbins = True )
        
        #bin_var = pd.qcut(dtn['P_ACT_Avg'],q = int(dbc['contin_to_n_bin']),retbins = True, labels = False)
        #bin_list =  bin_var[1][1:].tolist()
        bin_df[var[2]+'_bin'] = bin_var[1:]
        
        
        
        dtn_tv[var[3]+'_bin'], bin_var = pd.qcut(dtn_tv[var[3]],q = int(dbc['contin_to_n_bin']),labels = [1,2,3,4,5,6,7,8,9,10], retbins = True)
        
        #bin_var = pd.qcut(dtn['N_ROT_PLC_Avg_bin'],q = int(dbc['contin_to_n_bin']),retbins = True)
        #bin_list =  bin_var[1][1:].tolist()
        bin_df[var[3]+'_bin'] =  bin_var[1:]
        
        
        
        
        dtn_tv[var[5] + '_bin'], bin_var = pd.qcut(dtn_tv[var[5]],q = int(dbc['contin_to_n_bin']),labels = [1,2,3,4,5,6,7,8,9,10], retbins = True)
        
        #bin_var = pd.qcut(dtn['V_WIN_Avg_bin'],q = int(dbc['contin_to_n_bin']),retbins = True)
        #bin_list =  bin_var[1][1:].tolist()
        bin_df[var[5] + '_bin'] =  bin_var[1:]
        
        #bin_df.to_csv(param_io['dbc_directory'] + "table_bin.csv")
        
        ##### Binning on Holdout
        
        bin_list =  bin_df[var[0]+'_bin'].tolist()
        dtn_h[var[0]+'_bin'] =  np.where(dtn_h[var[0]]  <= bin_list[0],1,np.where(dtn_h[var[0]]  <= bin_list[1],2,np.where(dtn_h[var[0]]  <= bin_list[2],3,np.where(dtn_h[var[0]]  <= bin_list[3],4,np.where(dtn_h[var[0]]  <= bin_list[4],5,np.where(dtn_h[var[0]]  <= bin_list[5],6,np.where(dtn_h[var[0]]  <= bin_list[6],7,np.where(dtn_h[var[0]]  <= bin_list[7],8,np.where(dtn_h[var[0]]  <= bin_list[8],9,10)))))))))
       
        bin_list =  bin_df['oper_temp_rise_bin'].tolist()
        dtn_h['oper_temp_rise_bin'] =  np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[0],1,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[1],2,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[2],3,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[3],4,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[4],5,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[5],6,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[6],7,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[7],8,np.where(dtn_h['oper_temp_rise_avg']  <= bin_list[8],9,10)))))))))
       
        bin_list =  bin_df[var[2]+'_bin'].tolist()
        dtn_h[var[2]+'_bin'] =  np.where(dtn_h[var[2]]  <= bin_list[0],1,np.where(dtn_h[var[2]]  <= bin_list[1],2,np.where(dtn_h[var[2]]  <= bin_list[2],3,np.where(dtn_h[var[2]]  <= bin_list[3],4,np.where(dtn_h[var[2]]  <= bin_list[4],5,np.where(dtn_h[var[2]]  <= bin_list[5],6,np.where(dtn_h[var[2]]  <= bin_list[6],7,np.where(dtn_h[var[2]]  <= bin_list[7],8,np.where(dtn_h[var[2]]  <= bin_list[8],9,10)))))))))
       
        bin_list =  bin_df[var[3]+'_bin'].tolist()
        dtn_h[var[3]+'_bin'] =  np.where(dtn_h[var[3]]  <= bin_list[0],1,np.where(dtn_h[var[3]]  <= bin_list[1],2,np.where(dtn_h[var[3]]  <= bin_list[2],3,np.where(dtn_h[var[3]]  <= bin_list[3],4,np.where(dtn_h[var[3]]  <= bin_list[4],5,np.where(dtn_h[var[3]]  <= bin_list[5],6,np.where(dtn_h[var[3]]  <= bin_list[6],7,np.where(dtn_h[var[3]]  <= bin_list[7],8,np.where(dtn_h[var[3]]  <= bin_list[8],9,10)))))))))
       
        bin_list =  bin_df[var[5]+'_bin'].tolist()
        dtn_h[var[5]+'_bin'] =  np.where(dtn_h[var[5]]  <= bin_list[0],1,np.where(dtn_h[var[5]]  <= bin_list[1],2,np.where(dtn_h[var[5]]  <= bin_list[2],3,np.where(dtn_h[var[5]]  <= bin_list[3],4,np.where(dtn_h[var[5]]  <= bin_list[4],5,np.where(dtn_h[var[5]]  <= bin_list[5],6,np.where(dtn_h[var[5]]  <= bin_list[6],7,np.where(dtn_h[var[5]]  <= bin_list[7],8,np.where(dtn_h[var[5]]  <= bin_list[8],9,10)))))))))
       
        
        dtn =  pd.concat([dtn_tv, dtn_h], axis  = 0, ignore_index = True)
        dtn.sort_values(['Device','TimeStamp'],ascending  =[True, True], inplace = True)
        
        
        if not os.path.exists(param_io['dbc_directory']+mac+"/"+param_prep['target']+ "_" + dy+"_"+mnth+"_"+yr  +"/"):
            os.makedirs(param_io['dbc_directory']+mac+"/"+param_prep['target']+ "_" + dy+"_"+mnth+"_"+yr  +"/")
        bin_df.to_csv(param_io['dbc_directory']+mac+"/"+param_prep['target']+ "_" + dy+"_"+mnth+"_"+yr  +"/"+ "table_bin.csv")
        
        
        if skip_tagging == 'FALSE':
            dbc['train'] = 'TRUE'
            print("\n****** CALLING PREP DBC :: TRAIN ******* \n")
            dt1_trn = prep_dbc(dtn[dtn['train_val']=='t'], dbc, param_io, param_prep,mac,sep_model)
            dbc['train'] = 'FALSE'
            print("\n****** CALLING PREP DBC :: VALID ******* \n")
            dt1_val = prep_dbc(dtn[dtn['train_val'] !='t'], dbc, param_io, param_prep,mac,sep_model)
            dt1 =  pd.concat([dt1_trn,dt1_val], axis = 0,ignore_index = True)
        else:
            dbc['train'] = 'FALSE'
            print("\n****** CALLING PREP DBC :: TEST ******* \n")
            dt1 = prep_dbc(dtn, dbc, param_io, param_prep,mac,sep_model)
   
    else:
         dt1 = dtn
    dt1.sort_values(['Device','TimeStamp'], ascending = [True, True], inplace = True)

    if skip_tagging == 'FALSE':
        dt1_trn =None
        dt1_val = None
    if use_dbc == 'TRUE':    
        print("\n****** CREATING Z TRANSFORM COLUMNS ******* \n")
        for i in range(len(dbc['dvar']['ivar'])):
            i_vars = dbc['dvar']['ivar'][i]
            dt1['dbc_z_'+i_vars] = (dt1[dbc['target']] - dt1['dbc_mean_' + i_vars])/dt1['dbc_stdev_' + i_vars]
    
    if skip_tagging == 'FALSE':
        print("\n****** WRITE NEW DATASET TO CSV AND FEATHER FILES ******* \n")
        if param_io['skip_feather'] =='FALSE':
            file_name_d1 = param_io['working_directory']+mac+"_ee_d1.feather"
            if len(glob.glob(file_name_d1)) >0 :
                print("Skipping write feather")
                os.remove(file_name_d1)
                feather.write_dataframe(dt1, file_name_d1)
            else:
                feather.write_dataframe(dt1, file_name_d1)
            
 # Sandeep changed this to if instead of commenting the outut of csv
        else:  
            file_name_d1 = param_io['working_directory']+mac+"_ee_d1.csv"
            if len(glob.glob(file_name_d1)) >0 :
                os.remove(file_name_d1)
                dt1.to_csv(file_name_d1)
                print("Skipping write csv")
                dt1.to_csv(param_io['working_directory'] +mac+"_ee_d1.csv")
        
    
    prep_d1 = {}
    prep_d1['param_prep'] = param_prep
    prep_d1['dt1']       =        dt1

    return prep_d1

def mean_absolute_percentage_error(y_true, y_pred):
        y_true = check_array(y_true)
        y_pred = check_array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 


###### Main Code Begins ######
    
##### This function sort of collates the entire code process and encapuslates the modelling and output generation part.
#### Additional functionality  like R Square, Correlation generation are included in this part.

def model_main(dt,i, all_machine,param_prep,param_model,dbc, param_io):
    
    
    i = i
    all_machine = all_machine 
    param_prep   = param_prep
    param_model =  param_model
    dbc    =       dbc
    param_io =  param_io
    
    sep_model = config['PARAM MODEL']['sep_model']
    if sep_model == 'TRUE' and all_machine == 'FALSE':
        mac = config['PARAM MODEL']['mac_name']
        dt = dt[dt['Device'] ==  mac]
        i =  mac
    
    print("\n****** FILTER ABNORM PROCESS ******* \n")
    
    
    dt = ee_filter_abnorm(dt,param_prep)
    
    print(dt.shape)
    
    
    np.random.seed(param_model['rand_seed'])
    dt['rand'] = np.random.uniform(0,1,dt.shape[0])
    dt['train_val'] =""
    dt['train_val'] = np.where(dt['TimeStamp'].str[:7]>= param_model['holdout_ref_date'],'h',"" )
    dt_val_trn = dt[dt['train_val']!= 'h']
    dt_val_trn['train_val'] = np.where(dt_val_trn['rand']<param_model['train_perc'], 't',np.where(dt_val_trn['rand']<param_model['valid_perc'],'v',""))
    dt_hld = dt[dt['train_val']=='h']
    dt = pd.concat([dt_hld,dt_val_trn], ignore_index = True)

    
    
    
    
    target_mean =  np.mean(dt[dbc['target']][dt['train_val'] == 't']) #### Global mean for the entire dataset being used
    target_std   = np.std(dt[dbc['target']][dt['train_val'] == 't'])  #### Global std for the entire dataset being used
    
    dbc['target_global_mean'] = target_mean
    
    dbc['target_global_stdev']  = target_std
    
    #### Save the Mean and Std ####
    
    if sep_model == "TRUE" and all_machine == "TRUE":
        
        metric = []
        metric.append(target_mean)
        metric.append(target_std)
        
        
    else:
        mean_lst = []
        mean_lst.append(target_mean)
        std_lst  = []
        std_lst.append(target_std) 
        metric = pd.DataFrame({'MACHINE': i,'MEAN': mean_lst,'STDEV': std_lst})

    
    
    dtn = dt
    
    
    
    dtn.sort_values(['Device','TimeStamp'], ascending  = [True,True], inplace = True)
    
    
    print("\n****** STARTING EE_PRED_D1 ******* \n")
    skip_tagging = 'FALSE'
    prep_d1_time_start = time.time()
    prep_d1 =    ee_prep_d1(dtn, param_io, param_prep, dbc, param_model,skip_tagging,i, sep_model)  
    prep_d1_time_end =   time.time()
    prep_d1_time_dur =   prep_d1_time_end - prep_d1_time_start
    print("Total process run time for EE_PREP_D1 function is ",prep_d1_time_dur,"seconds")  
    param_prep = prep_d1['param_prep']
    dt1  =         prep_d1['dt1']
    print(dt1['train_val'].value_counts())
    
    
    
            
    reload_prep_data =  'FALSE'
    print("\n****** RELOAD THE NEW DATASET WITH NEW FEATURES ******* \n")
    if reload_prep_data == 'TRUE':
        if param_io['skip_feather'] == 'FALSE':
            file_name_d1 = param_io['working_directory'] + i+"_ee_d1.feather"
            dt1 = feather.read_dataframe(file_name_d1)
            
        else:
            dt1 = pd.read_csv(param_io['working_directory'] + i+"_ee_d1.csv")
            
    
    
    param_model['dbc_cols_chosen'] = config['PARAM MODEL']['dbc_cols_chosen'].split(",")
    
    
    param_model['cols_chosen'] = config['PARAM MODEL']['cols_chosen'].split(",")
    
    
    
    use_dbc =  config['PARAM MODEL']['use_dbc']
    if use_dbc == 'TRUE':
        param_model['input_cols'] = param_model['dbc_cols_chosen'] + param_model['cols_chosen']
    else:
        
        param_model['input_cols'] = param_model['cols_chosen']
    
    ####### XG Boost Modelling #########
    
    
    
    d1_trn_xgb_targ = dt1[param_model['target']][dt1['train_val']=='t']
    d1_val_xgb_targ = dt1[param_model['target']][dt1['train_val']=='v']
    d1_hld_xgb_targ = dt1[param_model['target']][dt1['train_val']=='h']
    
    
    input_col_txt = param_model['input_cols']
    
    d1_trn_xgb_inp = dt1[input_col_txt][dt1['train_val']== 't']
    d1_val_xgb_inp = dt1[input_col_txt][dt1['train_val']== 'v']
    d1_hld_xgb_inp = dt1[input_col_txt][dt1['train_val']== 'h']
    
    
    d1_trn_xgb_inp = d1_trn_xgb_inp.as_matrix()
    d1_val_xgb_inp = d1_val_xgb_inp.as_matrix()
    d1_hld_xgb_inp = d1_hld_xgb_inp.as_matrix()
    
    
    d1_trn_xgb = xgb.DMatrix(d1_trn_xgb_inp,label = d1_trn_xgb_targ)
    d1_val_xgb = xgb.DMatrix(d1_val_xgb_inp,label = d1_val_xgb_targ)
    d1_hld_xgb = xgb.DMatrix(d1_hld_xgb_inp,label = d1_hld_xgb_targ)
    
    
    
    
    param = {'max_depth':param_model['max_depth'], 'eta':param_model['eta'],'objective': param_model['objective']}
    num_round = param_model['num_round']
    
    print("\n****** MODEL TRAINING ******* \n")
    
    bst = xgb.train(param, d1_trn_xgb, num_round)
    
    

    #xgb.plot_importance(bst, color =  'red')
    
    
    run_time = datetime.datetime.now()
    
    if sep_model == 'FALSE':
        pickle.dump(bst, open(param_io['working_directory']+dbc['target']+str(run_time.year)+"_"+str(run_time.month)+"_"+str(run_time.day)+"_xgbmodel.pickle.dat", "wb"))
    else:
        pickle.dump(bst, open(param_io['working_directory']+dbc['target']+str(run_time.year)+"_"+str(run_time.month)+"_"+str(run_time.day)+"_"+i+"_xgbmodel.pickle.dat", "wb"))
    
    #### Predictions ####
    
    print("\n****** PREDICTION ******* \n")
    d1_trn_xgb_m2 = bst.predict(d1_trn_xgb)
    d1_val_xgb_m2 = bst.predict(d1_val_xgb)
    d1_hld_xgb_m2 = bst.predict(d1_hld_xgb)
    
    d1_trn_xgb_m2_list = d1_trn_xgb_m2.tolist()
    d1_val_xgb_m2_list = d1_val_xgb_m2.tolist()
    d1_hld_xgb_m2_list = d1_hld_xgb_m2.tolist()
    dt1_t = dt1[dt1['train_val'] == 't']
    dt1_v =dt1[dt1['train_val'] == 'v']
    dt1_h =dt1[dt1['train_val'] == 'h']
    
    dt1_t['d1_xgb_m2_predict'] = d1_trn_xgb_m2_list
    dt1_v['d1_xgb_m2_predict'] = d1_val_xgb_m2_list
    dt1_h['d1_xgb_m2_predict'] = d1_hld_xgb_m2_list
    
    
    dt1 = pd.concat([dt1_t,dt1_v, dt1_h], axis = 0)
        
        
    dt1['d1_xgb_m2_err'] = dt1[param_model['target']] - dt1['d1_xgb_m2_predict']
    
    
    
    ############### Save The Validation Error mean and std ##################
    
    if sep_model == 'TRUE' and all_machine == 'TRUE':
        
       mean_std_val = []
        
       mean_var = np.mean(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'v'])
       std_var = np.std(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'v'])
        
       mean_std_val.append(mean_var)
       mean_std_val.append(std_var)
    
    else:
        mean_val = []
        mean_val.append(np.mean(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'v']))
        std_val  = []
        std_val.append(np.std(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'v'])) 
        mean_std_val = pd.DataFrame({'MACHINE': i,'VALIDATION_MEAN': mean_val,'VALIDATION_STDEV': std_val})
    
    ###### Calculation of MachineWise Date Wise Sumof 'd1_xgb_m2_err'
    
    
    
    
    dt1_h = dt1[dt1['train_val'] == 'h']
    dt1_h['TimeStamp'] = pd.to_datetime(dt1_h['TimeStamp'])
    dt1_h['Date'] =  [dt.date() for dt in dt1_h['TimeStamp']]
    
    daily_df = dt1_h.groupby(['Device','Date'])['d1_xgb_m2_err'].agg(['sum']).reset_index()
    
    daily_df.columns = ['Machine','Date','Sum_Daily_Error']
    dt1_h = None
    
    if sep_model == 'TRUE' and all_machine == 'TRUE':
        
    
        daily_df.to_csv(param_io['output_directory'] + dbc['target']+i  + "_MachineDate_SumofError_holdout.csv", index =  False)
        
    elif sep_model == 'TRUE' and all_machine == 'FALSE':
        
        daily_df.to_csv(param_io['output_directory'] + dbc['target']+mac  + "_MachineDate_SumofError_holdout.csv", index =  False)
        
    else:
        daily_df.to_csv(param_io['output_directory'] +dbc['target'] +"_MachineDate_SumofError_holdout.csv", index =  False)
        
        
    ##### Output minimal fields for plotting #####
    
    print("\n****** PLOTTING DATA GENERATED ******* \n")
    score_out_const = ["Device","TimeStamp",dbc['target'],"d1_xgb_m2_predict","d1_xgb_m2_err","train_val"]
    score_out_var = config['PARAM MODEL']['score_out_var'].split(",")
    score_out_list = score_out_const + score_out_var
    score_out = dt1[score_out_list]
    score_out.to_csv(param_io['model_directory']+param_prep['target']+"_"+i+"_d1_xgb_m2_predict.csv")
    
    print("\n****** DATA WITH PREDICTIONS, RESIDUAL GENERATED ******* \n")
    if param_io['skip_feather'] == 'FALSE':
      feather.write_dataframe(dt1,param_io['model_directory']+ i + dbc['target']+"_dt1_xgb_m2.feather")
    else:  
     dt1.to_csv(param_io['model_directory']+ i + dbc['target']+ "_dt1_xgb_m2.csv")
    
    ##### Correlations between target and predicted ######
    
    ##1
    
    
    
    print(np.corrcoef(dt1[param_model['target']][dt1['train_val']=='t'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='t']))
    print(100*r2_score(dt1[param_model['target']][dt1['train_val']=='t'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='t']))
    print(np.corrcoef(dt1[param_model['target']][dt1['train_val']=='v'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='v']))
    print(100*r2_score(dt1[param_model['target']][dt1['train_val']=='v'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='v']))
    print(np.corrcoef(dt1[param_model['target']][dt1['train_val']=='h'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='h']))
    print(100*r2_score(dt1[param_model['target']][dt1['train_val']=='h'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='h'])) 
    
    if sep_model == 'TRUE' and all_machine == 'TRUE':
        rsq = 100*r2_score(dt1[param_model['target']][dt1['train_val']=='h'],dt1['d1_xgb_m2_predict'][dt1['train_val']=='h'])
    else:
        turbine_list = list(set(dt1.Device))
        rs_val =[]
        for i in turbine_list:
            dt_holdout =  dt1[dt1['train_val']== 'h']
            rs = 100*r2_score(dt_holdout[param_model['target']][dt_holdout['Device']== i ],dt_holdout['d1_xgb_m2_predict'][dt_holdout['Device']== i ])
            rs_val.append(rs)
        rsq_df1 = pd.DataFrame({'Machines': turbine_list,'RSQUARE': rs_val})
        rsq =  rsq_df1
            
    print(dt1[param_model['target']].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    print(dt1['d1_xgb_m2_err'][dt1['train_val'] == 't'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    print(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'v'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    print(dt1['d1_xgb_m2_err'][dt1['train_val'] == 'h'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    
    return rsq,metric,mean_std_val
    
    
################### For loop for multi machine and all machine model  #######################

log_level = 3
fwrite_prob = 'TRUE'

### Call define_param_io

param_io = define_param_io()
param_prep = define_param_prep()
use_full_data = 'TRUE'
param_model = define_param_model(use_full_data)   
dbc = define_param_dbc()


print("\n****** WRITE DBC TO CSV FILE ******* \n")

dbc['dvar'].to_csv(param_io['dbc_directory'] +"dbc_input_params.csv")


##### Loop through multiple intupt files and concatenate and create a single dataframe
st = time.time()
if param_io['skip_feather'] == 'FALSE':
    first_read = 'TRUE'
    if first_read == 'TRUE':
        
        
        if param_io['file_no'] == 1:
            #dt = pd.read_csv(param_io['file_loc'], usecols = param_prep['cols_read'])
            dt = feather.read_dataframe(param_io['file_loc'],columns = param_prep['cols_read'])
            
        elif param_io['file_no'] >= 2:
            dt = pd.DataFrame()
            for i in range(param_io['file_no']):
                print("\n****** READING ******* \n","FILE:",i+1)
                print("Reading File",param_io['file_loc'][i])
                #dummy_dt = pd.read_csv(param_io['file_loc'][i], usecols = param_prep['cols_read'])
                dummy_dt = feather.read_dataframe(param_io['file_loc'][i], columns = param_prep['cols_read'])
                print(dummy_dt.shape)
                dt = pd.concat([dt,dummy_dt],axis = 0,ignore_index = True)
                
        
        feather_file = param_io['input_directory']+"InputData.feather"
        
            
        feather.write_dataframe(dt,feather_file)
    else:
         print("\n******* FEATHER FILE PRESENT******* \n")
        
        
else:
    if param_io['file_no'] == 1:
            dt = pd.read_csv(param_io['file_loc'], usecols = param_prep['cols_read'])
    
    elif param_io['file_no'] >= 2:
        dt = pd.DataFrame()
        for i in range(param_io['file_no']):
            print("\n****** READING ******* \n","FILE:",i+1)
            print("Reading File",param_io['file_loc'][i])
            dummy_dt = pd.read_csv(param_io['file_loc'][i], usecols = param_prep['cols_read'])
            dt = pd.concat([dt,dummy_dt],axis = 0,ignore_index = True)
            
        
et = time.time()    
print("\n****** FILE READ TIME ******* \n", et-st,"seconds")


if param_io['skip_feather'] == 'FALSE':
    import time
    start_time = time.time()
    feather_file = param_io['input_directory']+"InputData.feather"
    dt = feather.read_dataframe(feather_file)
    end_time = time.time()
    elapsed_time = end_time- start_time
    print("Time Taken to read feather data ",elapsed_time," seconds")

all_machine = config['PARAM MODEL']['all_machine']
sep_model = config['PARAM MODEL']['sep_model']
mac_name =  config['PARAM MODEL']['mac_name']

##### If we are building multiple models say for Pitch Current then ideally we should have sep_model = TRUE and all_machine = TRUE




if sep_model == 'TRUE' and all_machine == 'TRUE':
    mac_list  = list(set(dt.Device))
    abnorm_mac_list = config['FILTER ABNORM']['remove_machine'].split(",")
    rsq_list =[]
    machine_list = []
    mean_list = []
    std_list  = []
    mean_val_list = []
    std_val_list   = []
    for i in mac_list:
        
        if i not in abnorm_mac_list:
            machine_list.append(i)
            dt_mac = dt[dt['Device'] == i]
            rsq,metric,mean_std_val = model_main(dt_mac,i,all_machine,param_prep,param_model,dbc, param_io)
            
            mean_val_list.append(mean_std_val[0])
            std_val_list.append(mean_std_val[1])            
            rsq_list.append(rsq)
            mean_list.append(metric[0])
            std_list.append(metric[1])
            print(i)
elif sep_model == 'TRUE' and all_machine == 'FALSE':
    dt_mac = dt[dt['Device'] ==  mac_name]
    rsq,metric,mean_std_val =  model_main(dt_mac,mac_name,all_machine,param_prep,param_model,dbc, param_io)
    
else:
    i = 'ALL'
    rsq,metric,mean_std_val = model_main(dt,i,all_machine,param_prep,param_model,dbc,param_io)



copyfile(config_path,param_io['working_directory']+dbc['target'] + yr+"_"+mnth+"_"+dy+"_"+"config.ini")

#### Creating The R Square File


if sep_model == 'TRUE' and all_machine == 'TRUE':
    rsq_df =  pd.DataFrame({'Machine':machine_list,'RSquare':rsq_list})
    rsq_df.to_csv(param_io['model_directory'] +yr+"_"+mnth+"_"+dy+"_" +dbc['target']+"_RSQUARE_HOLDOUT.csv")
    metric_df =  pd.DataFrame({'MACHINE': machine_list,'MEAN': mean_list,'STD': std_list})
    metric_df.to_csv(param_io['model_directory'] +yr+"_"+mnth+"_"+dy+"_" +dbc['target']+ "_MeanStdDF.csv") 
    pd.DataFrame({'MACHINE': machine_list, 'VALIDATION_MEAN':mean_val_list,'VALIDATION_STD': std_val_list}).to_csv(param_io['model_directory'] + yr + "_" + mnth + "_" + dy+ "_"+ dbc['target'] + "_validation_mean_std.csv")
else:
    rsq.to_csv(param_io['model_directory'] +yr+"_"+mnth+"_"+dy+"_" +dbc['target']+"_RSQUARE_HOLDOUT.csv")
    metric.to_csv(param_io['model_directory'] +yr+"_"+mnth+"_"+dy+"_" +dbc['target']+ "_MeanStdDF.csv")
    mean_std_val.to_csv(param_io['model_directory'] +yr+"_"+mnth+"_"+dy+"_" +dbc['target']+ "_validation_mean_std.csv")

####################   Scoring on Test Data   #####################
#################### The Config File disables the operation of this code. There is a seprate code for Prediction On Test Data



skip_test =  config['TEST']['skip_test']
multi_month = config['TEST']['multi_month']

if skip_test == 'FALSE':
    print("\n****** PREDICTION ON TEST DATA ******* \n")
    
    file_pattern = config['TEST']['data_pattern']
    file_loc_test = glob.glob(config['PATH']['test_data_dir']+file_pattern+"*.csv")
    file_no = len(file_loc_test)
    if file_no == 1:
        
        test_data = pd.read_csv(file_loc_test[0], usecols = param_prep['cols_read'])
        #test_data = test_data[param_prep['cols_chosen']]
    else:
        test_data = pd.DataFrame()
        for i in range(file_no):
            
            dummy_dt = pd.read_csv(file_loc_test[i], usecols = param_prep['cols_read'])
            test_data = pd.concat([test_data,dummy_dt],axis = 0,ignore_index = True)
        #test_data =  test_data[param_prep['cols_chosen']]
#    if config['TEST']['multi_month'] == 'FALSE':
#        month = config['TEST']['month']
#        test_data = test_data[test_data['TimeStamp'].apply(lambda x: x[0:7])== month]
#    else:
#        start_month = config['TEST']['start_month']
#        end_month  = config['TEST']['end_month']
#        test_data = test_data[(test_data['TimeStamp'].apply(lambda x: x[0:7])>= start_month) and (test_data['TimeStamp'].apply(lambda x: x[0:7])<= end_month)]
    print("\n****** FILTERING ABNORM TEST ******* \n")        
    
    if sep_model == 'TRUE':
        test_mac = config['TEST']['mac']
        i =  test_mac
        test_data = test_data[test_data['Device'] == test_mac]
    else:
        i = 'ALL'
    test_dtn = ee_filter_abnorm(test_data,param_prep)
    test_dtn.shape
    
    
    
    
    print("\n****** STARTING EE_PRED_D1: TEST ******* \n")
    test_data = None
    test_dtn.sort_values(['Device','TimeStamp'], ascending  = [True,True], inplace = True)
    skip_tagging = 'TRUE'
    prep_d1 =    ee_prep_d1(test_dtn, param_io, param_prep, dbc, param_model,skip_tagging,i, sep_model)  
    
    
    input_col_txt  =  param_model['input_cols']
    test_dt1 = prep_d1['dt1']
    dt1_test_xgb_targ = test_dt1[param_model['target']]
    dt1_test_xgb_inp = test_dt1[input_col_txt]
    dt1_test_xgb_inp = dt1_test_xgb_inp.as_matrix()
    
    model_year = config['TEST']['model_year']
    model_month = config['TEST']['model_month']
    model_day = config['TEST']['model_day']
    
    
    model_date = model_year + "_" + model_month + "_" +  model_day
    
    if sep_model == 'FALSE':
        print(dbc['target']+model_date)
        bst = pickle.load(open(param_io['working_directory']+dbc['target']+model_date+"_xgbmodel.pickle.dat", "rb"))
    else:
        print(dbc['target']+model_date+"_"+i)
        bst = pickle.load(open(param_io['working_directory']+dbc['target']+model_date+"_"+test_mac+"_xgbmodel.pickle.dat", "rb"))
        
    
    
    
    dt1_test_xgb = xgb.DMatrix(dt1_test_xgb_inp,label = dt1_test_xgb_targ )
    dt1_test_xgb_m2 = bst.predict(dt1_test_xgb)
    dt1_test_xgb_m2_list = dt1_test_xgb_m2.tolist()
    test_dt1['xgb_m2_predict'] = dt1_test_xgb_m2_list
    test_dt1['xgb_m2_err'] = test_dt1[param_model['target']] - test_dt1['xgb_m2_predict']
    
    ###### MAPE and RSQ  MACHINE WISE ######
    
    dev_list = list(set(test_dt1['Device']))
    mape_var = []
    rsq =[]
    for i in dev_list:
        x = mean_absolute_percentage_error(list(test_dt1[dbc['target']][test_dt1['Device'] == i].reshape(-1,1)),list(test_dt1['xgb_m2_predict'][test_dt1['Device'] == i].reshape(-1,1)) )
        y = 100*r2_score(test_dt1[dbc['target']][test_dt1['Device'] == i], test_dt1['xgb_m2_predict'][test_dt1['Device'] == i])
        mape_var.append(x)
        rsq.append(y)
    
    score_df = pd.DataFrame({'Device':dev_list,'MAPE':mape_var,'R_SQUARE':rsq})
    print(np.corrcoef(test_dt1[param_model['target']], test_dt1['xgb_m2_predict']))
    print("\n****** SCORED DATA: WRITE TO CSV ******* \n")
    
    test_dt1.to_csv(param_io['output_directory']+dbc['target']+"_predictedOutput.csv")
    score_df.to_csv(param_io['output_directory']+dbc['target']+"_MachineWiseScores.csv")

os.chdir(config['PATH']['home_dir'])
proc_end_time = time.time()

print("Time taken for code to Run: ",proc_end_time - proc_st_time)



###### Describe functions havent been included as of now , will be done if required to do so #######

###### Plotting Codes are omitted as per instruction #########


