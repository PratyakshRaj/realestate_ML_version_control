#modular code with exception handling 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import json

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}:{e}")

#test_data=pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data:pd.DataFrame)-> tuple[pd.DataFrame,pd.Series]:
    try:
      x=data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
      y=data["price"]
      return x,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}") 
#x_test= test_data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
#y_test=test_data["price"]

def load_model(filepath):
    try:
        with open(filepath,"rb") as file:
          model=pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
    

#model=pickle.load(open("model.pkl","rb"))

def evaluation_model(model,x_test,y_test):
    try:
        y_pred=model.predict(x_test)


        r2_score_=r2_score(10**y_test,10**y_pred)
        root_mse=((mean_squared_error(10**y_test,10**y_pred))**0.5)

        metrics_dict={
            "r2_score":r2_score_,
            "root_mean_square_error":root_mse
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")

def save_metrics(metrics_dict,filepath):
    try:
        with open(filepath,"w") as file:
            json.dump(metrics_dict,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}") 

def main():
    try:
        test_data_path="./data/processed/test_processed.csv"
        model_path="models/model.pkl"
        metrics_path="reports/metrics.json"
        
        test_data=load_data(test_data_path)
        x_test,y_test=prepare_data(test_data)
        model=load_model(model_path)
        metrics=evaluation_model(model,x_test,y_test)
        save_metrics(metrics,metrics_path)     
    except Exception as e:
        raise Exception(f"An error occured:{e}")
    
if __name__=="__main__":
    main()               
            