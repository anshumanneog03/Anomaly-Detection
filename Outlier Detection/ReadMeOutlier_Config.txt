#### ReadMe File for OutlierDetection Code ####

The outlier_detect code is meant to generate Residual Sums observation/daily level
wise for the input turbine data. It takes the Error variable(difference between the actual and 
predicted) and performs standardiazation,thresholding and applying cumulative sum function to come up with the Residual Sum. 

Additionally it also creates a day on day sum of the Z_risk_net_error to be used 
for future reference. Another additional functionality is, it plots the selected
machines/device i.e Residual Sum against Time.

outlier_detect_v0.8_wip_cumsum_bounded.py code is targeted at those output feature models which take all machines
to create their model i.e which have single model ex- T_BEAR_SHAFT_Avg.

LoopPlottingCodes_v0.5.py  code is for the current and power output features which have multi machine models

Config_Outlier Description::

[PATH]

input_directory 

mention the path of the folder containing the input data files.
This folder and the correpsonding file is created during the modelling/test/scoring exercise.


filename

mention the filename, which is to be picked up ex - ALL_dt1_xgb_m2.feather
(this is only for single model output features)

output_directory 

the output file path where all the outlier detection code outputs are to be saved

[OTHER]

holdout_ref_date

this date is to be mentioned in a year - month format and signifies the code to 
only take those data which are post this period i.e holdout

target 


the target variable or the output feature

err_var

the column name containing the error values

device_list 

to be only used for single models as they contain all the machines in the dataset
the names of the device are to be mentioned with only the number suffix like
001,002,003 etc
These devices are later taken in the code and only plots for these devices are created

z_risk_threshold

user defined limit

z_risk_drain_threshold

user defined limit

model_year =  mention of year of the model you are using
model_month = mention month of the model you are using
model_day  =  mention day of the model you are using


first_run = 
this is a flag variable, which ensures if the code is to be run on a holdout or test scored mode


##### Using the same config file for the Looping Outlier Detect Code #####


LoopPlottingCodes.py is used for the same purpose as above but it is used for multi machine 
models. Hence the list of machines have been hardcoded in the python script. This code picks up the 
machines in a loop creates the z_risk_net and residual_sum outputs and additonally the residual sum vs time
plot.


[OTHER]

device_list 

enter as per requirement

