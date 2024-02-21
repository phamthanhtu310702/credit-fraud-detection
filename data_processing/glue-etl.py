import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import concat_ws, to_json, struct
from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import DropFields, SelectFields
import pyspark.sql.functions as fc
import pyspark.pandas as ps
from neptune_python_utils.glue_gremlin_csv_transforms import GlueGremlinCsvTransforms
## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

def get_features_and_labels(transactions_df, transactions_id_cols, transactions_cat_cols):
    # Get features
    non_feature_cols = ['isFraud', 'TransactionDT'] + transactions_id_cols.split(",")
    feature_cols = [col for col in transactions_df.columns if col not in non_feature_cols]
    features = transactions_df.select(feature_cols) # pyspark.sql.df
    features = features.to_pandas_on_spark()

    kdf_features = ps.get_dummies(features, columns = transactions_cat_cols.split(",")).fillna(0) #pyspark.pandas.df
    features = kdf_features.to_spark()
    
    features = features.withColumn('TransactionAmt', fc.log10(fc.col('TransactionAmt')))
    
#     Get labels
    labels = transactions_df.select('TransactionID', 'isFraud')

    return features, labels

def dump_df_to_s3(df, objectName, header = True,output_prefix = 's3://fraud-transaction-detection/test_graph/' ,graph = False):
    if graph == False:
        objectKey = f"{output_prefix}{objectName}"
    else:
        objectKey = f"{output_prefix}graph/{objectName}"
    glueContext.write_dynamic_frame.from_options(
        frame=DynamicFrame.fromDF(df, glueContext, f"{objectName}DF"),
        connection_type="s3",
        connection_options={"path": objectKey},
        format_options={"writeHeader": header},
        format="csv")   

def get_relations_and_edgelist(transactions_df, identity_df, transactions_id_cols):
    # Get relations
    edge_types = transactions_id_cols.split(",") + list(identity_df.columns)
    new_id_cols = ['TransactionID'] + transactions_id_cols.split(",")
    full_identity_df = transactions_df.select(new_id_cols).join(identity_df, on='TransactionID', how='left')

    # extract edges
    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[['TransactionID', etype]].dropna()
        edges[etype] = edgelist
    return edges
    
def get_relations_and_edgelist2(transactions_df, transactions_id_cols):
    # Get relations
    edge_types = transactions_id_cols.split(",")
    new_id_cols = ['TransactionID'] + transactions_id_cols.split(",")
    full_identity_df = transactions_df.select(new_id_cols)

    # extract edges
    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[['TransactionID', etype]].dropna()
        edges[etype] = edgelist
    return edges

def dump_edge_as_graph(name, dataframe):
    # upsert edge
    dynamic_df = DynamicFrame.fromDF(dataframe, glueContext, f'{name}EdgeDF')
    relation = GlueGremlinCsvTransforms.create_prefixed_columns(dynamic_df, [('~from', 'TransactionID', 't'),('~to', name, name)])
    relation = GlueGremlinCsvTransforms.create_edge_id_column(relation, '~from', '~to')
    relation = SelectFields.apply(frame = relation, paths = ["~id", '~from', '~to'], transformation_ctx = f'selection_{name}')
    dump_df_to_s3(relation.toDF(), f'relation_{name}_edgelist', graph = True)

def dump_vertex_to_s3(name, dataframe):
    dynamic_df = DynamicFrame.fromDF(dataframe, glueContext, f'{name}VertexDF')
    vertex = GlueGremlinCsvTransforms.create_prefixed_columns(dynamic_df, [('~id', name, name)])
    vertex = GlueGremlinCsvTransforms.addLabel(vertex, name)
    vertex = SelectFields.apply(frame = vertex, paths = ["~id",'~label'])
    dump_df_to_s3(vertex.toDF(), f'vertex_{name}')
    
    

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

id_cols ='card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain' 
cat_cols ='M1,M2,M3,M4,M5,M6,M7,M8,M9'

transactions = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://fraud-transaction-detection/fraud-data/transaction_debug.csv"]},
    format="csv",
    format_options={
        "withHeader": True,
        # "optimizePerformance": True,
    },
)


features_df, labels_df = get_features_and_labels(transactions.toDF(), id_cols, cat_cols)
featurs_graph_df = features_df.withColumn('props_values:String', to_json(struct(list(filter(lambda x: (x != 'TransactionID'), features_df.schema.names)))))
featurs_graph_df = featurs_graph_df.select('TransactionID','props_values:String')

features_graph_dynamic_df = DynamicFrame.fromDF(featurs_graph_df, glueContext, 'FeaturesDF')
features_graph_dynamic_df = GlueGremlinCsvTransforms.create_prefixed_columns(features_graph_dynamic_df, [('~id', 'TransactionID', 't')])
features_graph_dynamic_df = GlueGremlinCsvTransforms.addLabel(features_graph_dynamic_df,'Transaction')
features_graph_dynamic_df = SelectFields.apply(frame = features_graph_dynamic_df, paths = ["~id",'~label', 'props_values:String'])
dump_df_to_s3(features_graph_dynamic_df.toDF(), f'transaction', graph = True)

relational_edges = get_relations_and_edgelist2(transactions.toDF(), id_cols)
for name, df in relational_edges.items():
    if name != 'TransactionID':
        dump_df_to_s3(df, f'relation_{name}_edgelist')
        dump_edge_as_graph(name, df)
        dump_vertex_to_s3(name,df[[name]])
job.commit()