import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from glib.metrics import get_node_classification_metrics
from glib.graph_loader import NaiveHetDataLoader
from ignite.utils import convert_tensor
import torch.nn.functional as F

def evaluate_model_node_classification(g , input_emb: nn.Module , model: nn.Module, criterion, neighbor_sampler: NaiveHetDataLoader, device):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = [], [], []
        for batch in neighbor_sampler:
            encoded_seeds, encoded_ids, edge_ids = batch
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

            evaluate_total_loss.append(loss.item())

            evaluate_y_trues.append(y_label)
            evaluate_y_predicts.append(predicts)


        evaluate_total_loss = np.mean(evaluate_total_loss)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics