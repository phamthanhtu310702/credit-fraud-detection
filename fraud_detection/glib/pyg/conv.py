from typing import Any, Dict, List, Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

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

        x =x + node_type_emb
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