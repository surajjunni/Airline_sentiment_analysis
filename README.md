# Airline Sentimant Analysis
Implementing an Machine Learning model to predict the emotion and deploying the model to server

## Overview
This software uses SVM model for predicting the results for sentimant analysis data.Later on,the model is deployed to a server with FastAPI where API endpoint is created.

## Environment
* Python 3.8
* nltk
* sklearn
* Pickle
* Numpy
* FastApi
* Pandas

## Running the Code
First script will train the model based on the dataset.
To run the script:
>python3 main.py

Second script is to intiate a API endpoint for the model.
To run the script:
>uvicorn app:app --reload


