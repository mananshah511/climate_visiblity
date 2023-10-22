import os,sys
from climate.logger import logging
from climate.exception import ClimateException
from climate.config.configuration import Configuration
from climate.entity.artifact_entity import DataIngestionArtifact
from climate.components.data_ingestion import DataIngestion

class Pipeline:

    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ClimateException(sys,e) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ClimateException(sys,e) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise ClimateException(sys,e) from e