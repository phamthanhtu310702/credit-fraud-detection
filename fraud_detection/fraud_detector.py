import logging
import shutil
import os
from glib.brisk_utils import create_naive_het_graph_from_edges
import pandas as pd
from parse_edgefile import construct_df
from glib.graph_loader import NaiveHetDataLoader, NaiveHetGraph
from glib.graph_loader import NaiveHetDataLoader
import torch
import numpy as np
from ignite.utils import convert_tensor
from glib.feature_engineering import FeatureEmbeddinng
from glib.fstore import FeatureStore
from glib.pyg.model import GNN, HetNet as Net
import torch.nn as nn
from glib.utils import EarlyStopping
from glib.metrics import get_node_classification_metrics
from glib.evaluate_model_utils import evaluate_model_node_classification
import torch.nn.functional as F
import yaml

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def test_graph():
    edge_files = pd.read_csv('files.txt',header=None)
    df, fn_mapping = construct_df(edge_files)
    labels = pd.read_csv('../processed_data/tags.csv')
    
    #assumption ts for experiment
    df['ts'] = 1
    g = create_naive_het_graph_from_edges(df, labels)
    shape = df[['dst', 'dst_type']].drop_duplicates().shape
    dst_id = df[['dst', 'dst_type']].drop_duplicates()
    dst_id['dst_id'] = range(1, 1 + shape[0])
    df = pd.merge(df, dst_id, how='inner', on =['dst', 'dst_type'])
    df = df.drop(['dst_id_x'], axis=1)
    df = df.rename(columns={'dst_id_y': 'dst_id' })
    label = pd.read_csv('../processed_data/tags.csv')
    g = create_naive_het_graph_from_edges(df, label)
    return g, fn_mapping

def main(path_g = './data/g_publish.parquet', path_feat_db='data/store.db', path_result='exp_result.csv',
         dir_model='./model',
         conv_name='het-emb', sample_method='sage',
         batch_size=(64, 16),
         width=16, depth=6,
         n_hid=400, n_heads=8, n_layers=6, dropout=0.2,
         optimizer='adamw', clip=0.25,
         n_batch=32, max_epochs=10, patience=8,
         seed_epoch=True, num_workers=0,
         seed=2020, debug=False, continue_training=False,
         non_blocking = False):
    
    graph_metadata = {}
    edge_files = pd.read_csv('files.txt',header=None)
    df, fn_mapping = construct_df(edge_files)
    labels = pd.read_csv('../processed_data/tags.csv')
    
    #assumption ts for experiment

    shape = df[['dst', 'dst_type']].drop_duplicates().shape
    dst_id = df[['dst', 'dst_type']].drop_duplicates()
    dst_id['dst_id'] = range(1, 1 + shape[0])
    df = pd.merge(df, dst_id, how='inner', on =['dst', 'dst_type'])
    df = df.drop(['dst_id_x'], axis=1)
    df = df.rename(columns={'dst_id_y': 'dst_id' })
    
    df_ts = df[['src']].drop_duplicates()
    df_ts['ts'] = range(1, 1 + df_ts.shape[0])
    df = pd.merge(df, df_ts, how='inner', on =['src'])

    label = pd.read_csv('../processed_data/tags.csv')
    g = create_naive_het_graph_from_edges(df, label)

    times = pd.Series(df['ts'].unique())
    times_train_valid_split = times.quantile(0.7)
    times_valid_test_split = times.quantile(0.9)
    train_range = set(t for t in times
                          if t is not None and t <= times_train_valid_split)
    valid_range = set(t for t in times
                          if t is not None and times_train_valid_split < t <= times_valid_test_split)
    test_range = set(t for t in times
                         if t is not None and t > times_valid_test_split)

    all_node_types = list(g.node_type_encode.keys())
    cat_node_types = list(fn_mapping.keys())
    store = FeatureStore(path='../data/feat_store_publish.db')
    dl_train = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=train_range, time_interval= (0, times_train_valid_split) ,method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=seed_epoch, num_workers=num_workers, shuffle=True)
    
    dl_val = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=valid_range, time_interval= (times_train_valid_split, times_valid_test_split),method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=seed_epoch, num_workers=num_workers, shuffle=True)
    
    dl_test = NaiveHetDataLoader(
            width=width, depth=depth,
            g=g, ts_range=test_range, time_interval= (times_valid_test_split, df['ts'].max()),method=sample_method,
            batch_size=batch_size, n_batch=n_batch,
            seed_epoch=seed_epoch, num_workers=num_workers, shuffle=True)
    
    #set up model
    num_feat = 389
    # num_feat = 389 // 371
    num_node_type = len(g.node_type_encode)
    num_edge_type = len(g.edge_type_encode)
    gnn = GNN(conv_name=conv_name,
                          n_in=n_hid,
                          n_hid=n_hid, n_heads=n_heads, n_layers=n_layers,
                          dropout=dropout,
                          num_node_type=num_node_type,
                          num_edge_type=num_edge_type
                          )
    model = Net(gnn, n_hid, num_embed=n_hid, n_hidden=n_hid)
    model.to(device)

    #set up feature embedding for all node types
    input_emb = FeatureEmbeddinng(graph= g, all_node_types= all_node_types, cat_node_types= cat_node_types, fn_mapping= fn_mapping, hidden_size= n_hid, fstore=store)
    params = list(model.parameters()) + list(input_emb.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params, lr= 0.001)


    model_name = 'fraud_detection'
    save_model_name = f'node_classification_{model_name}'
    save_model_folder = f"./saved_models/{model_name}/{save_model_name}/"

    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    early_stopping = EarlyStopping(patience=20, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, model_name=model_name)
    
    graph_metadata['node_type_encoder'] = g.node_type_encode
    graph_metadata['num_node_type'] = num_node_type
    graph_metadata['num_edge_type'] = num_edge_type
    graph_metadata['cat_node_types'] = cat_node_types
    graph_metadata['fn_mapping'] = fn_mapping
    graph_metadata['edge_type_encode'] = g.edge_type_encode
    with open('graph_metadata/graph_metadata.yaml', 'w') as file:
        yaml.safe_dump(graph_metadata, file)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1):
        train_total_loss, train_y_trues, train_y_predicts = [], [], []

        for batch in dl_train:
            optimizer.zero_grad()
            encoded_seeds, encoded_ids, edge_ids = batch
            encoded_ids = set(encoded_ids)
            encoded_seeds = set(encoded_seeds)
            encode_to_new = dict((e,i) for i, e in enumerate(encoded_ids))
            mask = np.asanyarray([e in encoded_seeds for e in encoded_ids])
            decoded_ids = [g.node_decode[e] for e in encoded_ids]

            #get initial feature_embedding
            initial_feature_embedding = input_emb(decoded_ids)
            initial_feature_embedding = torch.stack(initial_feature_embedding, dim=0)
            initial_feature_embedding= torch.FloatTensor(initial_feature_embedding)
            x = convert_tensor(torch.FloatTensor(initial_feature_embedding), device=device, non_blocking=False)
            
            edge_list = [g.edge_list_encoded[:, idx] for idx in edge_ids]
            f = lambda x: encode_to_new[x]
            f = np.vectorize(f)
            edge_list = [f(e) for e in edge_list]
            edge_list = [convert_tensor(torch.LongTensor(e), device=device, non_blocking=False) 
                        for e in edge_list]

            # becasue we sample duplicate seed node it exist in node_ids
            y = np.asarray([-1 if e not in encoded_seeds else g.seed_label_encoded[e] for e in encoded_ids])
            assert (y >= 0).sum() == len(encoded_seeds)
            y = torch.FloatTensor(y)

            y = convert_tensor(y, device=device, non_blocking=False)
            mask = torch.BoolTensor(mask)
            mask = convert_tensor(mask, device=device, non_blocking=False)

            y = y[mask]
            
            node_type_encode = g.node_type_encode
            node_type = [node_type_encode[g.node_type[e]] for e in decoded_ids]
            node_type = torch.LongTensor(np.asarray(node_type))
            node_type = convert_tensor(
                node_type, device=device, non_blocking=False)
            
            edge_type = [[g.edge_list_type_encoded[eid] for eid in list_] for list_ in edge_ids]
            edge_type = [torch.LongTensor(np.asarray(e)) for e in edge_type]
            edge_type = [convert_tensor(e, device=device, non_blocking=False) for e in edge_type]

            input, y_label = ((mask, x, edge_list, node_type, edge_type), y)
            logits = model(input)
            predicts =  logits.sigmoid().to(device)
            
            loss = criterion(logits, y_label)


            train_total_loss.append(loss.item())
            train_y_trues.append(y_label)
            train_y_predicts.append(predicts)

            loss.backward()
            optimizer.step()

        train_total_loss = np.mean(train_total_loss)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)
    
        train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)    
        
        val_total_loss, val_metrics = evaluate_model_node_classification(g= g, input_emb=input_emb,
                                                                        model=model,
                                                                        criterion = criterion,
                                                                        neighbor_sampler=dl_val,
                                                                        device= device)
        print(f'train loss: {train_total_loss:.4f}')     
        for metric_name in train_metrics.keys():
            print(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
        
        print(f'validate loss: {val_total_loss:.4f}')      
        for metric_name in val_metrics.keys():
            print(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

        val_metric_indicator = []
        for metric_name in val_metrics.keys():
            val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
        early_stop = early_stopping.step(val_metric_indicator, model)

        if early_stop:
            break
    
    input_embedding_name = 'input_embedding'
    save_model_path = os.path.join(save_model_folder, f"{input_embedding_name}.pkl")
    # how about the input_embedding layer ??
    torch.save(input_emb.state_dict(), save_model_path)
    early_stopping.load_checkpoint(model)
    test_total_loss, test_metrics = evaluate_model_node_classification(g= g, input_emb=input_emb,
                                                                        model=model,
                                                                        criterion = criterion,
                                                                        neighbor_sampler=dl_test,
                                                                        device= device)
    

    print(f'test loss: {test_total_loss:.4f}')
    for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            print(f'test {metric_name}, {test_metric:.4f}')

    # cache graph for reproducibility
    # build logger info
    # add ssl method to solve the imbalanced dataset
            
if __name__ == '__main__':
    main()