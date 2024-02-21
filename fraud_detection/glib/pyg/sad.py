import numpy as np
import torch
import torch.nn as nn
from glib.pyg.gdn import graph_deviation_network
from glib.pyg.supconloss import SupConLoss

class SAD(nn.Module):
    def __init__(self, cfg, device, node_feat_dim) -> None:
        super().__init__()
        self.gdn = graph_deviation_network(cfg, device, node_feat_dim)
        self.suploss = SupConLoss()

    def forward(self, source_node_embedding, node_ts, labels):
        anom_score = self.gdn(source_node_embedding, node_ts, labels)  # 异常检测
        dev, group = self.gdn.dev_diff(torch.squeeze(anom_score), node_ts)

        prediction_dict = {}
        prediction_dict['anom_score'] = anom_score
        prediction_dict['time'] = torch.from_numpy(node_ts)
        prediction_dict['group'] = group.clone().detach()
        prediction_dict['dev'] = dev.clone().detach()

        return prediction_dict

