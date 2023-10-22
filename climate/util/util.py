import os,sys,yaml
from climate.logger import logging
from climate.exception import ClimateException

def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ClimateException(sys,e) from e