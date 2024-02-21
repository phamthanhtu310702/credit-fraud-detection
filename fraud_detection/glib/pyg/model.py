import torch
import torch.nn as nn
from glib.pyg.conv import GeneralConv
from glib.pyg.modules import MLPClassifier


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

        source_node_embedding = self.out(x)
        x = self.affinity_score(x)
        logits  = torch.reshape(x, [-1])
        return logits  , source_node_embedding
