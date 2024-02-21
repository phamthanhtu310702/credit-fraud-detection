import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemakerSession = sagemaker.session.Session(boto3.session.Session(region_name=region))

class JSONPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, JSONSerializer, JSONDeserializer)

env = {
    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
    'SAGEMAKER_MODEL_SERVER_TIMEOUT' : '300'
}
model_data = 's3://fraud-transaction-detection/model.tar.gz'
fd_sl_model = PyTorchModel(image_uri = 'xxx.dkr.ecr.us-east-1.amazonaws.com/pytorch-gnn7:nvgpu',
                           name = 'inference-gnn-model',
                           model_data= model_data,
                           role=role,
                           entry_point='code/inference.py',
                           py_version='py3',
                           predictor_cls=JSONPredictor,
                           source_dir='code/',
                           env=env,
                           sagemaker_session=sagemakerSession)

predictor = fd_sl_model.deploy(instance_type='ml.c5.4xlarge', 
                                    initial_instance_count=1,
                                    endpoint_name = 'fraud-detection-gnn-inference'
                                    )
