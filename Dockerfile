# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-ec2

ENV PATH="/opt/ml/code:${PATH}"
# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Defines cifar10.py as script entrypoint 
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir multi-model-server sagemaker-inference
RUN pip install --no-cache-dir torch_geometric torch_scatter torch_sparse
RUN pip install --no-cache-dir pytorch-ignite