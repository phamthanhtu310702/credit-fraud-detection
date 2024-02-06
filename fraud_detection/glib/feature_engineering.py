# label encoding for cat type
import torch
from torch import nn
import numpy as np
class FeatureEmbeddinng(nn.Module):
    def __init__(self, graph ,all_node_types, cat_node_types, fn_mapping, hidden_size, fstore):
        super().__init__()
        # 2 type: continuous and categorical feature
        self.cat_embed = torch.nn.ModuleDict({
            node_type: torch.nn.Embedding(len(fn_mapping[node_type]), hidden_size) for node_type in all_node_types if node_type in cat_node_types
            })
        self.continuos_embed = torch.nn.ModuleDict(
            {node_type: nn.Linear(1, hidden_size) for node_type in all_node_types if node_type not in cat_node_types})
        self.transaction_embed = torch.nn.Linear(371, hidden_size)
        
        self.cat_node_types = cat_node_types
        self.graph = graph
        self.fn_mapping = fn_mapping
        self.fstore = fstore

    def forward(self, node_ids):
        # get node type form node_ids in Graph
        # node_mapping : dict {node type: {value: index}}
        node_embedding = []
        for node_id in node_ids:
            node_type = self.graph.node_type[node_id]
            if node_type != 'TransactionID': 
                node_value = self.graph.node_dst_value[node_id]

            # create a Dict node_raw_value if node_type is not Target_node
            # (node_ids, value)
            if node_type in self.cat_node_types:
                node_embedding.append(self.cat_embed[node_type](torch.tensor(self.fn_mapping[node_type][node_value])))
            elif node_type != 'TransactionID': 
                node_embedding.append(self.continuos_embed[node_type](torch.tensor([node_value], dtype=torch.float32)))
            elif node_type == 'TransactionID':
                 with self.fstore.db.begin() as wb:
                    node_embedding.append(self.transaction_embed(torch.tensor(self.fstore.get(key=int(node_id), default_value= np.zeros(389), wb=wb))))

        return node_embedding



