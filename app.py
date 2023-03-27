# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:48:08 2023

@author: Virupaksha H S
"""

import streamlit as st
from pycaret.regression import *
import pickle
import os
import sys
import requests


# Construct the relative file path to the CSV file
autoML_file = "taxi38_autoML_pipeline"

loaded_model = load_model(autoML_file)

# Create a function to predict the fare amount
def predict_fare(hour, day, month, source, destination, cab_type, name, distance, 
                 surge_multiplier, temperature, short_summary, precipIntensity, precipIntensityMax, precipProbability, humidity, windSpeed, 
                 windGust, visibility, temperatureHigh, temperatureLow, dewPoint, pressure, windBearing, cloudCover, uvIndex, ozone):
    input_dict = {'hour': [hour], 
                  'day': [day],
                  'month': [month],                  
                  'source': [source],
                  'destination': [destination],
                  'cab_type': [cab_type],
                  'name': [name],
                  
                  'distance': [distance],
                  'surge_multiplier': [surge_multiplier],
                  'temperature': [temperature],                  
                  'short_summary': [short_summary],
                  'precipIntensity': [precipIntensity],
                  'precipIntensityMax': [precipIntensityMax],
                  'precipProbability': [precipProbability],
                  'humidity': [humidity],
                  'windSpeed': [windSpeed],
                  'windGust': [windGust],
                  'visibility': [visibility],
                  'temperatureHigh': [temperatureHigh],
                  'temperatureLow': [temperatureLow],
                  'dewPoint': [dewPoint],
                  'pressure': [pressure],
                  'windBearing': [windBearing],
                  'cloudCover': [cloudCover],
                  'uvIndex': [uvIndex],
                  'ozone': [ozone],
                  }
    input_df = pd.DataFrame.from_dict(input_dict)
    return predict_model(loaded_model, data=input_df)['Label'][0]


# Define the Streamlit app
st.title('Taxi Fare Prediction')
st.write('Enter the details below to predict the taxi fare amount')

hour = st.slider('Hour', 0, 23, 9)
day = st.slider('Day', 1, 31, 23)
month = st.slider('Month', 1, 12, 5)
source = st.selectbox('Source', ['Boston University', 'Beacon Hill', 'Back Bay',  
                                 'Fenway', 'Financial District', 'Haymarket Square', 
                                 'North End', 'North Station', 'Northeastern University', 
                                 'South Station', 'Theatre District', 'West End'])
destination = st.selectbox('Destination', ['Back Bay', 'Beacon Hill', 'Boston University', 
                                           'Fenway', 'Financial District', 'Haymarket Square', 
                                           'North End', 'North Station', 'Northeastern University', 
                                           'South Station', 'Theatre District', 'West End'])
cab_type = st.selectbox('Cab Type', ['Lyft', 'Uber'])
name = st.selectbox('Cab name', ['Shared', 'Lux', 'Lyft', 'Lux Black XL', 'Lyft XL', 
                                 'Lux Black', 'UberXL', 'Black', 'UberX', 'WAV', 
                                 'Black SUV', 'UberPool', 'Taxi'])
# price = st.slider('price', 2.00, 100.00, 16.43)
distance = st.number_input('Distance (miles)', value=2.0, min_value=0.0, max_value=100.0, step=0.1, format='%f')
surge_multiplier = st.selectbox('surge multiplier', [1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
temperature = st.slider('Temperature', 0.00, 60.00, 39.52)
short_summary = st.selectbox('Weather summary', [' Mostly Cloudy ', ' Rain ', ' Clear ', 
                                                     ' Partly Cloudy ', ' Overcast ', ' Light Rain ', 
                                                     ' Foggy ', ' Possible Drizzle ', ' Drizzle '])
precipIntensity = st.slider('precipitation Intensity', 0.0000, 0.2000, 0.0089)
precipIntensityMax = st.slider('precipation Intensity Max', 0.0000, 0.2000, 0.0376)
precipProbability = st.slider('precipitation Probability', 0.0000, 0.1000, 0.1454)
humidity = st.slider('humidity', 0.30, 1.00, 0.74)
windSpeed = st.slider('wind Speed', 0.20, 20.00, 6.22)
windGust = st.slider('wind Gust', 0.50, 30.00, 8.51)
visibility = st.slider('visibility', 0.500, 15.000, 8.474)
temperatureHigh = st.slider('High temperature', 30.00, 60.00, 44.95)
temperatureLow = st.slider('Low temperature', 0.00, 58.00, 34.17)
dewPoint = st.slider('Dew Point', 3.00, 55.00, 31.57)
pressure = st.slider('pressure', 980.00, 1040.00, 1009.85)
windBearing = st.slider('wind Bearing', 0.00, 359.00, 220.25)
cloudCover = st.slider('Cloud Cover', 0.00, 1.00, 0.687)
uvIndex = st.selectbox('uvIndex', [0, 1, 2])
ozone = st.slider('ozone', 250.00, 400.00, 314.02)

# When the 'Predict' button is clicked, make the prediction
if st.button('Predict Fare'):
    fare = predict_fare(hour, day, month, source, destination, cab_type, name, distance, 
                        surge_multiplier, temperature, short_summary, precipIntensity, 
                        precipIntensityMax, precipProbability, humidity, windSpeed, windGust, 
                        visibility, temperatureHigh, temperatureLow, dewPoint, pressure, 
                        windBearing, cloudCover, uvIndex, ozone)
    st.success(f'The estimated fare amount is ${fare:.2f}')
    predict_fare()




if st.button("Refresh"):
    # Send a GET request to the URL to refresh the app
    requests.get("http://localhost:8501/")

