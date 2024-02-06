import os
import json
from typing import Tuple, Dict, List, Optional, Union, Set
import torch 
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import convert_tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
import numpy as np
import yaml 
import pandas as pd

BASE_PATH = '/opt/ml/model/code/'

device = torch.device('cpu')

class FeatureEmbeddinng(nn.Module):
    def __init__(self, all_node_types, cat_node_types, fn_mapping, hidden_size):
        super().__init__()
        # 2 type: continuous and categorical feature
        self.cat_embed = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(len(fn_mapping[node_type]), hidden_size) for node_type in all_node_types if node_type in cat_node_types
            })
        self.continuos_embed = torch.nn.ModuleDict(
            {node_type: nn.Linear(1, hidden_size) for node_type in all_node_types if node_type not in cat_node_types})
        self.transaction_embed = torch.nn.Linear(371, hidden_size)
        self.cat_node_types = cat_node_types
        self.fn_mapping = fn_mapping


    def forward(self, node_ids, node_types, node_dst_value , n_feats):
        # get node type form node_ids in Graph
        # node_mapping : dict {node type: {value: index}}
        node_embedding = []
        for node_id in node_ids:
            node_type = node_types[node_id]
            if node_type != 'TransactionID': 
                node_value = node_dst_value[node_id]

            # create a Dict node_raw_value if node_type is not Target_node
            # (node_ids, value)
            if node_type in self.cat_node_types:
                node_embedding.append(self.cat_embed[node_type](torch.tensor(self.fn_mapping[node_type][node_value])))
            elif node_type != 'TransactionID': 
                node_embedding.append(self.continuos_embed[node_type](torch.tensor([node_value], dtype=torch.float32)))
            elif node_type == 'TransactionID':
                node_embedding.append(self.transaction_embed(torch.tensor(np.array(n_feats.loc[node_id].values).astype('float32'))))

        return node_embedding

class HetEmbConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True,
                 num_node_type=-1, num_edge_type=-1,
                 edge_type_self=0, **kwargs):
        super().__init__(node_dim=0, aggr='add', **kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        assert num_node_type > 0
        assert num_edge_type > 0
        self.edge_type_self = edge_type_self
        self.lin_q = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = Linear(in_channels, heads * out_channels, bias=False)
        self.node_type_embeddings = nn.Embedding(num_node_type, in_channels)
        self.edge_type_embeddings = nn.Embedding(num_edge_type, in_channels)

        self.reset_parameters()
    def reset_parameters(self):
        # glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.bias)
        glorot(self.lin_q.weight)
        glorot(self.lin_k.weight)
        glorot(self.lin_v.weight)
        self.node_type_embeddings.reset_parameters()
        self.edge_type_embeddings.reset_parameters()
    def forward(self, x, edge_index, node_type, edge_type):
        node_type_emb = self.node_type_embeddings(node_type)

        x = x + node_type_emb
        xq = self.lin_q(x)
        x = (x, x)
        xq = (xq, xq)
        
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]
        
        #self_loops
        len_no_self = edge_index.shape[1]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))
        edge_type_self = torch.ones(edge_index.shape[1]-len_no_self).long() * self.edge_type_self
        edge_type_self = edge_type_self.to(edge_index.device)
        edge_type = torch.cat([edge_type, edge_type_self])
        
        
        edge_emb = self.edge_type_embeddings(edge_type)
        out = self.propagate(edge_index, x=x, xq=xq, edge_emb=edge_emb)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def message(self, xq_i, x_j, edge_index_i, size_i, 
                edge_emb):
        # Compute attention coefficients.
        xq_i = xq_i.view(-1, self.heads, self.out_channels)

        x_j += edge_emb
        x_k = self.lin_k(x_j)
        x_k = x_k.view(-1, self.heads, self.out_channels)

        x_v = self.lin_v(x_j)
        x_v = x_v.view(-1, self.heads, self.out_channels)

        alpha = (xq_i * self.att_i).sum(-1) + (x_k * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, None, size_i)
        # alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_v * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout, 
                 num_node_type, num_edge_type, edge_index_ts):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'het-emb':
            self.base_conv = HetEmbConv(
                in_hid, out_hid // n_heads, heads=n_heads, dropout=dropout, 
                edge_type_self=0, 
                num_node_type=num_node_type, num_edge_type=num_edge_type)
        else:
            raise NotImplementedError('unknown conv name %s' % conv_name)
    def forward(self, x, edge_index, node_type, edge_type, *args):
        if self.conv_name in {'het-emb'}:
            return self.base_conv(x, edge_index, node_type, edge_type)
        raise NotImplementedError('unknown conv name %s' % self.conv_name)
    
class GNN(nn.Module):
    def __init__(self, n_in, n_hid, n_layers, n_heads, dropout, 
                 conv_name, num_node_type, num_edge_type,
                 edge_index_ts=None):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.conv_name = conv_name

        self.gcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        if conv_name in ['het-emb']:
            self.init_fc = nn.Linear(n_in, n_hid)
            for i in range(n_layers):
                layer = GeneralConv(
                    conv_name, 
                    in_hid=n_hid, 
                    out_hid=n_hid, n_heads=n_heads, dropout=dropout,
                    num_node_type=num_node_type, num_edge_type=num_edge_type, edge_index_ts=edge_index_ts)
                self.gcs.append(layer)
                layer = nn.LayerNorm(n_hid)
                self.norms.append(layer)
        
    def forward(self, x, edge_index, node_type, edge_type):
        #convert to the iteratable 
        if edge_type is None:
            edge_type = [None] * len(edge_index)
        if not isinstance(edge_index, (list, tuple)):
            edge_index = [edge_index]
        if not isinstance(edge_type, (list, tuple)):
            edge_type = [edge_type]

        if edge_type is None:
            edge_type = [None] * len(edge_index)

        assert len(edge_index) == len(edge_type)

        # duplicate edge index for each layer
        if len(self.gcs) > len(edge_index):
            if len(edge_index) == 1:
                edge_index = [edge_index[0]] * len(self.gcs)
                edge_type = [edge_type[0]] * len(self.gcs)
            else:
                raise RuntimeError(
                    'Mismatch layer number gcs %d and edge_index %d!' % (
                        len(self.gcs), len(edge_index)))
        # if self.init_fc:
        #     x = self.init_fc(x)
        for conv, norm, ei, et in zip(self.gcs, self.norms, edge_index, edge_type):
            x = conv(x, ei, node_type, et)
            x = self.dropout(x)
            x = norm(x)
            x = torch.relu(x)
        return x


class HetNet(nn.Module):
    def __init__(self, gnn, num_feature, num_embed,
                 n_hidden=128, num_output=2, dropout=0.5, n_layer=1):
        super(HetNet, self).__init__()
        if gnn is None:
            self.gnn = None
            self.fc1 = nn.Linear(num_feature, n_hidden)
        else:
            self.gnn = gnn
            self.fc1 = nn.Linear(2*num_feature, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.lyr_norm1 = nn.LayerNorm(normalized_shape=n_hidden)

        if n_layer == 1:
            pass
        elif n_layer == 2:
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.lyr_norm2 = nn.LayerNorm(normalized_shape=n_hidden)
        else:
            raise NotImplementedError()
        self.n_layer = n_layer

        self.out   = nn.Linear(n_hidden, n_hidden)
        self.affinity_score = MLPClassifier(n_hidden, dropout=0.2)
        
    def forward(self, x, edge_index=None, **kwargs):
        if edge_index is None:
            mask, x, edge_index, *args = x
        else:
            args = tuple()
            mask = torch.arange(x.shape[0])
        if self.gnn is not None:
            x0 = x
            x = self.gnn(x0, edge_index, *args, **kwargs)
            x = torch.tanh(x)
            x = torch.cat([x0, x], -1)

        x = x[mask]
        x = self.fc1(x)
        x = self.lyr_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        if self.n_layer == 1:
            pass
        elif self.n_layer == 2:
            x = self.fc2(x)
            x = self.lyr_norm2(x)
            x = torch.relu(x)
            x = self.dropout(x)
        else:
            raise NotImplementedError()

        x = self.out(x)
        x = self.affinity_score(x)
        x = torch.reshape(x, [-1])
        return x
   
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)

def initialize_arguments():

    with open('deployment/code/graph_metadata/graph_metadata.yaml', 'r') as file:
        graph_metadata = yaml.safe_load(file)
    etypes = graph_metadata['num_edge_type']
    ntype_dict = graph_metadata['num_node_type']

    input_size = 400
    hidden_size = 400
    n_layers = 6
    n_heads = 8
    drop_out = 0.2

    return ntype_dict, etypes, input_size, hidden_size, n_layers, n_heads, drop_out

def model_fn():
    
    ntype_dict, etypes, in_size, hidden_size, n_layers, n_heads, drop_out= initialize_arguments()
    # initialize_arguments(os.path.join(BASE_PATH, 'metadata.pkl'))

    gnn = GNN(conv_name='het-emb',
                          n_in=in_size,
                          n_hid=hidden_size, n_heads=n_heads, n_layers=n_layers,
                          dropout=drop_out,
                          num_node_type=ntype_dict,
                          num_edge_type=etypes
                          )
    model = HetNet(gnn, in_size, num_embed=hidden_size, n_hidden=hidden_size)
    model.load_state_dict(torch.load('deployment/saved_model/model.pkl'))

    return model.to(device)

def input_fn(request_body, request_content_type='application/json'):
    input_data = json.loads(request_body)
    
    subgraph_dict = input_data['graph']
    n_feats = input_data['n_feats']
    target_id = input_data['target_id']

    mask, x, edge_list, encoded_node_types, edge_list_type_encoded = generate_subgraph(subgraph_dict, n_feats, target_id)
    return mask, x, edge_list, encoded_node_types, edge_list_type_encoded


def generate_subgraph(graph_dict, n_feats, target_id):

    "n_feats: mapping TransactionId to the feature Transaction"
    "Ex: n_feats[TransactionId]"
    "subgraph_dict contoin pairs of nodes - edge"
    "create node_idx, edge_idx and then feed to model"
    "save a mapping function - Graphmetadata"

    with open('graph_metadata/graph_metadata.yaml', 'r') as file:
        graph_metadata = yaml.safe_load(file)
    
    node_type = {}
    node_dst_value = {}
    dest_id = 1
    edge_list = []

    new_n_feats = {}
    for i in n_feats.keys():
        new_n_feats[int(i)] = n_feats[i]
    
    ### process n_feats
    n_feats_df = pd.DataFrame.from_dict(new_n_feats, orient='index')
    n_feats_df = n_feats_df.fillna(0.0)
    n_feats_df = n_feats_df.replace('', 0)

    input_embedding = FeatureEmbeddinng(all_node_types= graph_metadata['node_type_encoder'].keys(),
                          cat_node_types= graph_metadata['cat_node_types'],
                          fn_mapping = graph_metadata['fn_mapping'],
                          hidden_size=400)
    input_embedding.load_state_dict(torch.load('deployment/code/input_embedding.pkl'))

    for transaction_id in new_n_feats.keys():
        node_type[transaction_id] = 'TransactionID'
    
    for can_etype, src_dst_tuple in graph_dict.items():

        src_type, dst_type = can_etype.split('<>')
        dst_origin = set(src_dst_tuple[1])
        
        for dst_node_value in dst_origin:
            node_type[dest_id] = dst_type
            node_dst_value[dest_id] = dst_node_value
            
            for src, dst in zip(src_dst_tuple[0],src_dst_tuple[1]):
                edge_list.append((src, dest_id, 'TransactionID<>' + dst_type))
            
            dest_id +=1

    node_encode = dict((n, i) for i, n in enumerate(node_type.keys()))
    all_nodes_ids = list(node_type.keys())
    
    initial_feature_embedding = input_embedding(all_nodes_ids, node_type, node_dst_value, n_feats_df)
    initial_feature_embedding = torch.stack(initial_feature_embedding, dim=0)
    initial_feature_embedding= torch.FloatTensor(initial_feature_embedding)
    x = convert_tensor(torch.FloatTensor(initial_feature_embedding), device=device, non_blocking=False)
   
    edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
    
    edge_list_encoded = np.zeros((2, len(edge_list)))
    for i, e in enumerate(edge_list):
        edge_list_encoded[:, i] = [node_encode[e[0]], node_encode[e[1]]]
    
    mask = np.asanyarray([e == target_id for e in all_nodes_ids])
    

    node_type_encoder = graph_metadata['node_type_encoder']
    
    encoded_node_types = [node_type_encoder[node_type[e]] for e in all_nodes_ids]
    encoded_node_types = torch.LongTensor(np.asarray(encoded_node_types))
    encoded_node_types = convert_tensor(
                encoded_node_types, device=device, non_blocking=False)
    
    edge_types = [a[2] for a in edge_list]

    edge_encode = graph_metadata['edge_type_encode']
    edge_list_type_encoded = [edge_encode[e] for e in edge_types]
    edge_list_type_encoded = torch.LongTensor(np.asarray(edge_list_type_encoded))
    edge_list_type_encoded = convert_tensor(edge_list_type_encoded, device=device, non_blocking=False)
    edge_list = convert_tensor(torch.LongTensor(edge_list_encoded), device=device, non_blocking=False)

    return (mask, x, edge_list, encoded_node_types, edge_list_type_encoded)



def predict_fn(input_data, model):

    mask, x, edge_list, encoded_node_types, edge_list_type_encoded = input_data

    with torch.no_grad():
        logits = model((mask, x, edge_list, encoded_node_types, edge_list_type_encoded))
        res = logits.sigmoid().cpu().detach().numpy()

    return res


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return prediction.cpu().numpy().tolist()
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")