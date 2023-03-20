#!/usr/bin/env python
# coding: utf-8

# # Dynamic Taxi Fare Prediction

import pandas as pd
from pycaret.regression import compare_models, predict_model
from pycaret.regression import *
import pandas_profiling
from pandas_profiling import ProfileReport
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys



# In Local Machine, Set the working directory to "Current working directory"
# os.getcwd()
# directory_path = os.path.join(os.getcwd(), 'taxiapp')

# From GitHUb repository, use the URL for the repo.
Dastaset_url = "https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma"
GitHub_url = r"https://github.com/Veeru-Hebbal/Dynamic-taxi-fare-prediction-using-taxirides-and-weather-variables"

# Construct the relative file path to the CSV file
data_file = os.path.join(GitHub_url, 'data', 'rideshare_kaggle_20k.csv')

# Load the dataset
data = pd.read_csv(data_file, nrows=20000)
# data.head()

data.nunique(axis=0)
# data.columns

# Statstical values for the every column as follows:
# data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
# data.info()

# Only 'price' column is having missing values.
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=["total", "percent"])
# missing_data.head()

# Let's check how many categorical and numerical columns are present in the data.
len(data._get_numeric_data().columns)

# We have total 46 columns as numeric data columns. Remaining 11 are categorical columns.
categorical_cols=data.columns[data.dtypes =='object']
# print(categorical_cols)
# categorical_cols.shape

# # Data Cleaning and Transformation
# We can remove the following columns, as they are not much helpful in providing insights.<br>
# [Id]: Used only for unique records.<br><br>
# [Datetime, Timestamp]: We already have month, day, hour variables, therfore these two could be removed.<br><br>
# [Timezone]: Only one timezone so we can remove it.<br><br>
# [Product_id]:The product_id could be removed, as we are more interested in product name.<br><br>
# [columns related to weather]. Currently not dealing with the weather aspect and hence removing all the related columns.<br><br>
# [longitude and latitude]: The source and destination names are availables, therefore these two could be removed.<br><br>
# data.info()

data = data.drop(['id', 'timestamp', 'datetime', 'timezone', 'product_id', 
                  'latitude', 'longitude', 'apparentTemperature',
                  'long_summary','windGustTime', 'temperatureHighTime', 'temperatureLowTime', 
                  'apparentTemperatureHigh', 'apparentTemperatureHighTime','apparentTemperatureLow', 
                  'apparentTemperatureLowTime', 'icon', 'visibility.1',  'sunriseTime', 'sunsetTime', 
                  'moonPhase','uvIndexTime', 'temperatureMin','temperatureMinTime', 'temperatureMax',
                  'temperatureMaxTime','apparentTemperatureMin', 'apparentTemperatureMinTime',
                  'apparentTemperatureMax', 'apparentTemperatureMaxTime'], axis=1)
# data.columns
# data.info()

# Imputing the mean value in place of missing values.
for j in ["price"]:
    data.loc[data.loc[:,j].isnull(),j] = data.loc[:,j].mean()
# data.price.isnull().sum()
data.isnull().sum()

# We can see that the outliers are present in the dataset. We can transform them by imputing mean values.
for i in ["price"]:
    data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].mean()
data.price.isnull().sum()

# Extracting categorical columns and numerical columns from the updated dataset.
categorical_cols=data.columns[data.dtypes =='object']
# print(categorical_cols)
# len(categorical_cols)

for i in categorical_cols:
  print(data[i].value_counts())

numeric_cols=data._get_numeric_data().columns
# print(numeric_cols)

data.isna().sum()

# dropping the NaN values from the dataset.
data = data.dropna(axis=0).reset_index(drop=True)

# taxiRides_dataset.info()    
data = data.fillna(0)
# data.info()    

# Set up the PyCaret environment
# Train and evaluate the model
reg = setup(data=data, target='price', session_id=773, normalize=True, 
                   transformation=True, transform_target=True, combine_rare_levels=True, 
                   rare_level_threshold=0.05, remove_multicollinearity=True, 
                   multicollinearity_threshold=0.95, log_experiment=True, 
                   experiment_name='taxi_weather_expr_03', profile=True, silent=True)

# All the process of model pipeline is given in each line below.
best_model = compare_models()
# print(best_model)

tuned_best = tune_model(estimator = best_model, optimize = 'R2')
# The datatype of the results in pycaret is not as a pandas DataFrame.
# type(tuned_best)

# To export the metrics results from 'pycaret' datatype to pandas 'DataFrame' using "pull()"
best_model_results = pull()
#type(best_model_results)

# NOTE: the get_logs() function works only if 'log_experiment = True' is selected during initializing the'setup'.
experiment_logs = get_logs()
#experiment_logs

# check the residuals of trained model
# plot_model(tuned_best, plot = 'residuals_interactive')

# check feature importance
# plot_model(tuned_best, plot = 'feature')
evaluate_model(tuned_best)

predicted_price = predict_model(tuned_best)

final_best = finalize_model(estimator = tuned_best)

# save the profiling report
design_report = ProfileReport(data)


autoEDA_file = os.path.join(GitHub_url, 'models', 'taxi38_autoEDA.html')
design_report.to_file(output_file = autoEDA_file)

# Saving the trained model
# Saving the model in .pkl file format.
autoML_file = os.path.join(GitHub_url, 'models', 'taxi38_autoML.pkl')
pickle.dump(final_best, open(autoML_file, "wb"))

# save the entire pipeline including the final model
autoML_pipeline = os.path.join(GitHub_url, 'models', 'taxi38_autoML_pipeline')
save_model(final_best, autoML_pipeline, model_only=False)

# Saving the model in .sav file format.
# import pickle
autoML_sav = os.path.join(directory_path, 'models', 'taxi38_autoML.sav')
pickle.dump(final_best, open(autoML_sav, 'wb')) # wb = writing binary

# data.info()
# data.columns

### END of the Script ###
