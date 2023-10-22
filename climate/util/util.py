import os,sys,yaml,dill
from climate.logger import logging
from climate.exception import ClimateException
import pandas as pd
from climate.constant import DROP_COLUMN_LIST

def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ClimateException(sys,e) from e
    
def write_yaml_file(file_path:str, data:dict=None):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        
        with open(file_path,"w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise ClimateException(sys,e) from e  
    
def load_object(file_path:str):
    try:
        with open(file_path,"rb") as object_file:
            return dill.load(object_file)
    except Exception as e:
        raise ClimateException(sys,e) from e
    
def preprocessing(df=pd.DataFrame):
    try:
        df.drop(DROP_COLUMN_LIST,inplace=True,axis=1)
        return df
    except Exception as e:
        raise ClimateException(sys,e) from e