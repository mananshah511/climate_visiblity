from flask import render_template,Flask,request
import os,sys,json
from flask_cors import CORS,cross_origin
from climate.logger import logging
from climate.exception import ClimateException
import pandas as pd
import numpy as np
from climate.pipeline.pipeline import Pipeline
from climate.entity.artifact_entity import FinalArtifact
from climate.util.util import load_object,preprocessing


app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    try:

        data = [x for x in request.form.values()]

        if not os.path.exists('data.json'):
            return render_template('index.html',output_text = "No model is trained, please start training")

        with open('data.json', 'r') as json_file:
            dict_data = json.loads(json_file.read())

        final_artifact = FinalArtifact(**dict_data)
        logging.info(f"final artifact : {final_artifact}")

        train_df = pd.read_csv(final_artifact.ingested_train_data)
        columns = list(train_df.columns)
        columns.pop(1)
        df = pd.DataFrame(data).T
        df.columns = columns
        df = preprocessing(df=df)

        df = (np.array(df.iloc[0])).reshape(1,-1)

        cluster_object = load_object(file_path = final_artifact.cluster_model_path)
        cluster_number = int(cluster_object.predict(df))

        model_object = load_object(file_path = final_artifact.export_dir_path[cluster_number])
        output = model_object.predict(df)
        return render_template('index.html',output_text = f"Visiblity distance is {output} kms")
    except Exception as e:
        raise ClimateException(sys,e) from e


@app.route('/train',methods=['POST'])
@cross_origin()
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        return render_template('index.html',prediction_text = "Model training completed")
    except Exception as e:
        raise ClimateException(sys,e) from e
    


if __name__ == "__main__":
    app.run()