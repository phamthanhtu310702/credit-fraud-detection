import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

import numpy as np
import tqdm
from glib.graph_loader import NaiveHetGraph
from glib.utils import timeit
import pandas as pd

def create_naive_het_graph_from_edges(df, labels):
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['src', 'ts']].drop_duplicates()
        
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        
        view = df[['src', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        view = df[['dst_id', 'dst_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))

        view = df[['dst_id', 'dst']].drop_duplicates()
        node_dst_value = dict(
            (node, value)
            for node, value in view.itertuples(index=False)
        )

    with timeit(logger, 'edge-list-init'):
        edge_list = list(
            df[['src', 'dst_id', 'edge_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]


    view = labels[['TransactionID', 'isFraud']].drop_duplicates()
    seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    return NaiveHetGraph(node_type, node_dst_value ,edge_list,
                         seed_label=seed_label,node_ts =node_ts)

def get_features(node_feature_files):
    """
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_features: path to file containing node features
    :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
    """
    df = pd.read_csv(node_feature_files)
    # node_feats = df.iloc[:, 1:].to_numpy()
    node_feats = df.iloc[:, 1:372].to_numpy()
    
    nodes_id = df.iloc[:,0]
    features = np.array(node_feats).astype('float32')
    return nodes_id, features
