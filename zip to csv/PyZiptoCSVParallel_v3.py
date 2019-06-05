
# This code processes the Zipped files and converts them into Wide Format

# Import requisite


import pandas as pd
import numpy as np
from datetime import datetime
import os 
import glob
import zipfile
from joblib import Parallel, delayed
import multiprocessing
from time import gmtime, strftime
import time
from multiprocessing import Pool
import configparser

os.getcwd()

#### Declaring Location of Zip Files

config = configparser.ConfigParser()
config.read('E:\Data2018may12_June12\zip_config.ini')

location = config['PATH']['location']
folder = config['PATH']['folder']

DataDir = ("/").join([location,folder])
os.chdir(DataDir)

filepattern = config['FILE']['filepattern']
fileext =     config['FILE']['fileext']

#get names of zipped files
if fileext == "csv":
    filenames = glob.glob(filepattern + "*.csv")
else:
    filenames = glob.glob(filepattern + "*.zip")
    
#get file numbers     
nfiles =  len(filenames)
nchannels = int(config['FILE']['nchannels'])


##############################

### Creating a dummy dataframe similar to the expected output

Tenmindata = pd.DataFrame(index = np.arange(0), columns = np.arange(nchannels*4+2))
if fileext == 'csv':
    Tenmindataraw = pd.read_csv(filenames[0])
else:
    Tenmindataraw = pd.read_csv(filenames[0], compression= config['FILE']['fileext'], header=0, sep=',', quotechar='"')
header_df  = pd.DataFrame(index = np.arange(0), columns = np.arange(nchannels*4+2))

Tenmindataraw.sort_values([config['FILE']['endtime'],'Tagname'], axis = 0, ascending = [True, True],inplace = True)
Tenmindataraw.reset_index(drop = True, inplace = True)
###############################

col_headings = ['TimeStamp','Device']
sensor = []
header_df.columns =header_df.columns.astype(str)
header_df.rename(columns={'0': col_headings[0],'1': col_headings[1]}, inplace=True)

###############################

for i in range(0,nchannels):
    
    channelnames = Tenmindataraw['Tagname'][i]
   
    chanstringsplit = channelnames.split(".")
    sensor.append(chanstringsplit[2])
    
    
    header_df.rename(columns={ header_df.columns[2 + nchannels*(1-1) + i]: str(sensor[i])+"_Avg" }, inplace = True)
    header_df.rename(columns={ header_df.columns[2 + nchannels*(2-1) + i]: str(sensor[i])+"_Min" }, inplace = True)
    header_df.rename(columns={ header_df.columns[2 + nchannels*(3-1) + i]: str(sensor[i])+"_Max" }, inplace = True)
    header_df.rename(columns={ header_df.columns[2 + nchannels*(4-1) + i]: str(sensor[i])+"_StdDev" }, inplace = True)
     
################################


colnames = header_df.columns
Tenmindata.columns = colnames

################################


    
    
######################### Parallel Processing #################################    

##### The code below structures the process in a functional way so that it can be used independently .. This also creates 
#### the scope of parallelization
##### Since the code is essentially in the wide format , that is each record defines a machine chracteristic like for ex- Power in Average, Min , Max and StdDev
#####  A  looping code is used to populate the long format data by having features such as Power_Avg, Power_Min, Power_Max and Power_Stddev


def readraw10minPara(rawfilename):
    rawfileext = config['FILE']['fileext']
    printme = True
    temp = None
    sensor =[]
    if printme == True:
        print("Filename=",rawfilename)
    if rawfileext == "csv":
        Tenmindataraw = pd.read_csv(rawfilename)
    else:
        Tenmindataraw = pd.read_csv(rawfilename, compression= config['FILE']['fileext'], header=0, sep=',', quotechar='"')

    Tenmindataraw.sort_values([config['FILE']['endtime'],'Tagname'], axis = 0, ascending = [True, True],inplace = True)
    Tenmindataraw.reset_index(drop = True, inplace = True)
       
    ndevices = int(config['FILE']['ndevices'])#Tenmindataraw.shape[0]/nchannels
    
    if ndevices!= int(config['FILE']['ndevices']):
        print("Number of channels is different than expected", ndevices)
    
    Tenmindata_df = pd.DataFrame(index = np.arange(ndevices), columns = np.arange(nchannels*4+2))
    Tenmindata_df.columns = colnames
    ndevices = int(ndevices)
    s= 0
    for nd in range(0,ndevices):
        Tenmindata_df['TimeStamp'] = Tenmindataraw['UTCTimestampEndPeriod']
        for nc in range(0,nchannels):
            
            channelname = Tenmindataraw['Tagname'][s]
            temp = channelname.split(".")
            sensor.append(temp[2])
            device = temp[1]
            Tenmindata_df.iloc[nd,1] = device
            Tenmindata_df.iloc[nd,2+nc] = Tenmindataraw['AverageValue'][s]
            Tenmindata_df.iloc[nd,2+1*nchannels+nc] = Tenmindataraw['MinValue'][s]
            Tenmindata_df.iloc[nd,2+2*nchannels+nc] = Tenmindataraw['MaxValue'][s]
            Tenmindata_df.iloc[nd,2+3*nchannels+nc] = Tenmindataraw['StdDev'][s]

            s = s + 1


    return Tenmindata_df   


    
##### The parallel processing in the main code uses multiple cores to perform the same job simultaneously for different Zip Files 

    
if __name__ == '__main__':
    with Pool() as p:
        start = time.time()
        result = p.map(readraw10minPara, filenames)    
        df = pd.concat(result)
        path =  config['OUTPUT']['path']
        filename =  config['OUTPUT']['filename']
        year = config['OUTPUT']['year']
        df.to_csv(path + filename+ year+'.csv')
        end = time.time()
        print(end - start)
