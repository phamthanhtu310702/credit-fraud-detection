import os
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import glob
import fire
import lmdb
from glib.fstore import FeatureStore 
from glib.brisk_utils import get_features
def main(path_db='../data/feat_store_publish.db'):
    
    store = FeatureStore(path=path_db)
    nodes_id, features = get_features('../processed_data/features.csv')
   
    with store.db.begin(write=True) as wb:
        for i in range(nodes_id.shape[0]):
            key = int(nodes_id[i])
            value = features[i]
            store.put(key, value, dtype=np.float32, wb=wb)

        # add average feats of neighbour txn nodes to entity nodes

    
    with store.db.begin() as wb:
        neighbor_feat = store.get(key=int(nodes_id[0]), default_value= np.zeros(371), wb=wb)
    print(neighbor_feat.shape)
    print(type(neighbor_feat))
if __name__=="__main__":
    fire.Fire(main)