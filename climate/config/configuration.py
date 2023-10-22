import os,sys
from climate.logger import logging
from climate.exception import ClimateException
from climate.constant import *
from climate.util.util import read_yaml
from climate.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformConfig

class Configuration:

    def __init__(self,config_file_path:str=CONFIG_FILE_PATH,
                 current_time_stamp:str=CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml(config_file_path)
            self.current_time_stamp = current_time_stamp
            self.training_pipeline_config = self.get_training_pipeline_config()
        except Exception as e:
            raise ClimateException(sys,e) from e
        
    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            logging.info(f"get data ingestion config function started")

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            data_ingestion_dir = os.path.join(artifact_dir,DATA_INGESTION_DIR,self.current_time_stamp)

            raw_data_dir = os.path.join(data_ingestion_dir,data_ingestion_config[DATA_INGESTION_RAW_DATA_DIR])

            ingested_data_dir = os.path.join(data_ingestion_dir,data_ingestion_config[DATA_INGESTION_INGESTED_DATA_DIR])

            ingested_train_data_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_INGESTED_TRAIN_DATA_DIR])

            ingested_test_data_dir = os.path.join(ingested_data_dir,data_ingestion_config[DATA_INGESTION_INGESTED_TEST_DATA_DIR])

            data_download_url = data_ingestion_config[DATA_INGESTION_DOWNLOAD_URL_KEY]

            data_ingestion_config = DataIngestionConfig(dataset_download_url=data_download_url,
                                                        raw_data_dir=raw_data_dir,
                                                        ingested_train_dir=ingested_train_data_dir,
                                                        ingested_test_dir=ingested_test_data_dir)

            logging.info(f"data ingestion config : {data_ingestion_config}")

            return data_ingestion_config
        except Exception as e:
            raise ClimateException(sys,e) from e

    def get_data_validation_config(self)->DataValidationConfig:
        try:
            logging.info(f"get data validation config function started")

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_config = self.config_info[DATA_VALIDTION_CONFIG_KEY]

            data_validation_dir = os.path.join(artifact_dir,DATA_VALIDATION_DIR,self.current_time_stamp)

            schema_dir = os.path.join(ROOT_DIR,data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                      data_validation_config[DATA_VALIDATION_SCHEMA_FILE_KEY])
            
            report_dir = os.path.join(data_validation_dir,data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME])

            data_validation_config = DataValidationConfig(schema_file_dir=schema_dir,
                                                          report_page_file_dir=report_dir,
                                                          report_name=data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME])
            
            logging.info(f"data validation config : {data_validation_config}")

            return data_validation_config
        except Exception as e:
            raise ClimateException(sys,e) from e
        
    def get_data_transform_config(self)->DataTransformConfig:
        try:
            logging.info(f"get data transform config function started")

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transform_config = self.config_info[DATA_TRANSFORM_CONFIG_KEY]

            data_transform_dir = os.path.join(artifact_dir,DATA_TRANSFORM_DIR,self.current_time_stamp)

            graph_dir = os.path.join(data_transform_dir,data_transform_config[DATA_TRANSFORM_GRAPH_DIR_KEY])

            transform_train_dir = os.path.join(data_transform_dir,data_transform_config[DATA_TRANSFORM_TRAIN_DIR_KEY])

            transform_test_dir = os.path.join(data_transform_dir,data_transform_config[DATA_TRANSFORM_TEST_DIR_KEY])

            cluster_model_dir = os.path.join(data_transform_dir,data_transform_config[DATA_TRANSFORM_CLUSTER_MODEL_DIR_KEY],
                                             data_transform_config[DATA_TRANSFORM_CLUSTER_MODEL_NAME_KEY])
            
            preprocessed_model_dir = os.path.join(data_transform_dir,data_transform_config[DATA_TRANSFORM_PREPROCESSED_OBJECT_DIR_KEY],
                                                  data_transform_config[DATA_TRANSFORM_PREPROCESSED_OBJECT_FILE_NAME_KEY])
            
            data_transform_config = DataTransformConfig(graph_save_dir=graph_dir,
                                                        transform_train_dir=transform_train_dir,
                                                        transform_test_dir=transform_test_dir,
                                                        cluster_model_file_path=cluster_model_dir,
                                                        preprocessed_file_path=preprocessed_model_dir)
            logging.info(f"data transform config: {data_transform_config}")

            return data_transform_config
        except Exception as e:
            raise ClimateException(sys,e) from e
        
        
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            logging.info(f"get training pipeline cofig function started")

            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]

            artifact_dir = os.path.join(ROOT_DIR,training_pipeline_config[TRINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)

            logging.info(f"training pipeline config : {training_pipeline_config}")

            return training_pipeline_config
        except Exception as e:
            raise ClimateException(sys,e) from e