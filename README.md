# Credit Card Fraud Detection
The emerging of online payment platforms brings much convenience in era of technological advancement. Besides, cyberthieves try to get the credit card information to make illegal transactions.
Deploying AI for fraud prevention has helped companies enhance their internal security and stream-
line business processes. Through improved efficiency, AI has emerged as an essential technology
to prevent fraud. This project applies Heterogeneous Graph Transformer neural network (HGT)
to compute risk score of each transaction. This model is trained on public Fraud Dataset and then
deployed to AWS for real-time inference

![](/images/diagram.png)

## Dataset
This project uses [the IEEE-CIS fraud dataset](https://www.kaggle.com/c/ieee-fraud-detection/data). The Dataset is originally in form of tableau  dataset. The dataset includes a transactions table and an identities table, having nearly 500,000 anonymized transaction records along with contextual information (for example, devices used in transactions). Some transactions have a binary label, indicating whether a transaction is fraudulent.

## Data preprocessing
### Training Stage
In training stage, we need to process tabular dataset into graph data, which includes edge files and transaction features, for reproducibilty of locally training model. 

After downloading the dataset, run the following command to process the dataset.


```
python data_processing/graph_processing.py --data-dir fraud_data --output-dir processed_data --id-cols card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain R_emaildomain --cat-cols M1,M2,M3,M4,M5,M6,M7,M8,M
```                                                                         

### Inference Stage
In inference stage, we need to process dataset by using AWS Glue and then upload processed  data to AWS S3. To access the graph and query supgraph in real-time infernence, We upload processed data into graph database - AWS Neptune.

## Fraud detection model
Inspired by the work from [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library/), the model in this project is built on top of Pytorch Geometric instead of Deep Graph Library. Besides, in the previous work the model is based on supervised learning and full graph training which means that the model is trained fully with labeled data and hard to be scaled when the dataset is large. So, this project applies the minibatch sampling technique in [GraphSage](https://arxiv.org/abs/1706.02216) which is inductive learning and able to scale. With the lack of labeled data and the affect of imbalanced dataset, the semi-supervised technique [GraphSAD](https://arxiv.org/abs/2305.13573) can make the model more robust.

GraphSage is an inductive learning which implies that it does not require the whole graph structure during learning and it can generalize well to the unseen nodes. GraphSAD is able to fully exploit the potential of large unlabeled samples and uncover underlying anomalies on evolving graph streams

The model backbone is [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) HGT instead of Heterogeneous Graph Neural Network in the previous work. To capture the heterogeneous relation patterns and learn more expressive node representations, a self-attentive heterogeneous graph neural network is adopted


## Model Training
Before training the model, we first set up Lightning Memory-Mapped Database LMDB, Which is a key-value store. LMDB uses memory-mapped files, giving much better I/O performance. It works well with large datasets.
We use LMDB to store the transaction features to reduce the data loading time.

```
python fraud_detection/setup_feature_store.py
```
Run ther following command to train the model:
```
python fraud_detection/fraud_detector.py
```
## Model Deployment
The deployment code is already [here](deployment/). This folder contains the deployment code. The graph_meta file, which is used for later inference, contains some necessary information about the trained model. The model is deployed on AWS Sagemaker endpoint for real-time inference.

## Model Inference
The inference code is [here](lambda/inferenceAPI.py). The code is fully copied from [this work](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl/blob/main/src/lambda.d/inference/func/inferenceApi.py). The lambda function tries to proccess the incoming transactions. And then, it inserts the new data into graph database Neptune and query the supgraph. Then the lambda function invokes Sagemaker endpoint and send the supgraph to the Sagemaker endpoint to compute the risk score of the transactions.

## References
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library/)
- https://arxiv.org/abs/2003.01332
- https://arxiv.org/abs/2305.13573
- https://arxiv.org/abs/1706.02216
- https://arxiv.org/abs/2011.12193