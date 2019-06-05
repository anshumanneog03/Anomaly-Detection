##### Code to Convert OSI String Fault Rows to NANs and generate Reports  #####
### Input : main datframe generated after collating the Zip Files ###
### Output: NullCount of each Feature 
###       : List of All string Columns
###       : Intermediate Data path with NANs at string errors
### Next Step : Removing the NAN rows


import numpy as np
import pandas as pd
import configparser
import os
import re
import glob
from dateutil.parser import parse
from datetime import timedelta
from sklearn import linear_model
import re

# Read config file 
config = configparser.ConfigParser()
config.read('E:\Data2018may12_June12\main_config.ini')
# 


### Read CSV File ###
os.getcwd()
directory = config['DIRECTORY PATHS']['STRINGROWREM_IDENTIFY_PATH']
write_path = config['DIRECTORY PATHS']['STRINGROWREM_IDENTIFY_WRITE_PATH']

if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

if not os.path.exists(write_path):
    os.makedirs(write_path)


user_decision = config['DROP COL']['UserDecision']
start_point = config['THRESHOLDS']['StartColumn']
start_point = int(start_point)
limit = config['THRESHOLDS']['Limit']
string_columns = config['STRING COLUMNS']['Names']
string_columns = string_columns.split(",")
time_columns = config['TIMESTAMP COLUMNS']['cols']
time_columns = time_columns.split(",")

filepattern = config['DIRECTORY PATHS']['filepattern']
listFiles = glob.glob('*.csv')
original_data = pd.read_csv(listFiles[0])

try:
    original_data.drop('Unnamed: 0', axis = 1,inplace = True)
except ValueError:
    pass


#### Converting the Time Column into TimeFormat  ####
for i in range(0, len(time_columns)):
    original_data[time_columns[i]] = pd.to_datetime(original_data[time_columns[i]])

#### Converting String type Variables to integer type  ####
###########################################################

colnames = original_data.columns.tolist()
colnames = colnames[start_point:]


str_list= []

### Gets the list of those columsn which are in a string format, indicates that there are Errors in these columns(Device nd TimeStamp omitted)
for i in range(0,len(colnames)):
    if type(original_data[colnames[i]][0]) == str:
        str_list.append(colnames[i])
string_df = pd.DataFrame({'ColName': str_list}) 
string_df.to_csv(write_path +'StringTypeColumnList.csv')                            
                              
#### extracts those columns which are string type only if they are not explicitly meant to remain as strings and converts the columns to numeric type

year =  config['DIRECTORY PATHS']['year']

for i in range(start_point,len(colnames)+start_point):
    if original_data.columns[i] not in string_columns:
        
        print("Identifying Column No "+ str(i))
        original_data[colnames[i-2]] = pd.to_numeric( original_data[colnames[i-2]],errors = 'coerce')
original_data.to_csv(write_path+filepattern+year+"BB2_10mindata_MayJune_nan.csv", index = False)       
import feather
original_data.to_feather(write_path+filepattern+year+"BB2_10mindata_MayJune_nan.feather")
### Get Count of NAs for each column ###

null_count = original_data.isnull().sum()  
null_count = null_count.to_frame() 
null_count[0] = null_count[0]/original_data.shape[0]
null_count['Columns'] = null_count.index 
null_count.columns.astype(str)
null_count.to_csv(write_path+'NullCount.csv', index = False)

