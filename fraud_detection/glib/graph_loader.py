# Copyright 2020-2021 eBay Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Dict, List, Optional, Union, Set
from collections import defaultdict
import logging
from functools import lru_cache
logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from torch_geometric.data import Data as PygData


import tqdm

from xfraud.glib.utils import timeit


class NaiveHetGraph(object):

    logger = logging.getLogger('native-het-g')

    def __init__(self, node_type: Dict[int, str], edge_list: Tuple[int, int, str],
                 seed_label: Dict[int, int], node_ts: Dict[int, int]):
        self.logger.setLevel(logging.INFO)
        self.node_type = node_type
        self.node_type_encode = self.get_node_type_encoder(node_type)        

        self.seed_label= seed_label
        self.node_ts = node_ts

        with timeit(self.logger, 'node-enc-init'):
            self.node_encode = dict((n, i) for i, n in enumerate(node_type.keys())) # encode to node_idx
            self.node_decode = dict((i, n) for n, i in self.node_encode.items())
        nenc = self.node_encode

        with timeit(self.logger, 'edge-type-init'):
            edge_types = [a[2] for a in edge_list]
            edge_encode = dict((v, i+1) for i, v in enumerate(set(edge_types)))
            edge_encode['_self'] = 0
            edge_decode = dict((i, v) for v, i in edge_encode.items())
            self.edge_type_encode = edge_encode
            self.edge_type_decode = edge_decode
            self.edge_list_type_encoded = [edge_encode[e] for e in edge_types]
        
        self.edge_list_encoded = np.zeros((2, len(edge_list)))
        for i, e in enumerate(tqdm.tqdm(edge_list, desc='edge-init')):
            self.edge_list_encoded[:, i] = [nenc[e[0]], nenc[e[1]]] #node_ids, node_idx

        with timeit(self.logger, 'seed-label-init'):
            self.seed_label_encoded = dict((nenc[k], v) for k, v in seed_label.items()) #idx, label

    def get_seed_nodes(self, ts_range) -> List:
        return list([e for e in self.seed_label.keys() 
            if self.node_ts[e] in ts_range])

    
    def get_node_type_encoder(self, node_type: Dict[int, str]):
        types = sorted(list(set(node_type.values())))
        return dict((v, i) for i, v in enumerate(types))

class NaiveHetDataLoader(object):
    
    logger = logging.getLogger('native-het-dl')

    def __init__(self, width: Union[int, List], depth: int, 
                 g: NaiveHetGraph, ts_range: Set, batch_size: int, n_batch: int, seed_epoch: bool, 
                 shuffle: bool, num_workers: int, method: str, cache_result: bool=False):

        self.g = g
        self.ts_range = ts_range

        if seed_epoch:
            batch_size = sum(batch_size)
            n_batch = int(np.ceil(len(self.seeds)/batch_size))
        else:
            assert len(batch_size) == len(self.label_seed)

        self.seed_epoch = seed_epoch
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.depth = depth
        self.width = width if isinstance(list, tuple) else [width] * depth
        assert len(self.width) == depth

        self.method = method
        self.cache_result = cache_result
        self.cache = None
    
    @property
    @lru_cache()
    def seeds(self):
        return self.g.get_seed_nodes(self.ts_range)
    
    @property
    @lru_cache()
    def label_seed(self)-> Dict[int,int]:
        seeds = self.seeds
        label_seed = defaultdict(list)
        # get seed for train/val/test from whole graph based on ts
        for sd, lbl in self.g.seed_label.items():
            if sd in seeds:
                label_seed[lbl].append(sd)

        return label_seed
    
    def __len__(self):
        return self.n_batch
    
    def sample_seeds(self) -> List:
        if self.seed_epoch:
            return self.g.get_seed_nodes(self.ts_range)
        # sample based-on batch size instead get all seeds
        rval =[]
        lbl_sd = self.label_seed
        for i,bz in enumerate(self.label_seed):
            cands = lbl_sd[i]
            rval.extend(
                np.random.choice(
                    cands, bz, replace=len(cands)<bz)
                )
        return rval
    def iter_sage(self):
        if self.cache_result and self.cache and len(self.cache) == len(self):
            self.logger.info('DL loaded from cache')
            for e in self.cache:
                yield e
        else:
            from torch.utils.data import DataLoader
            g = self.g
            seeds = self.sample_seeds()
            seeds_encoded = [g.node_encode[e] for e in seeds]
            sampler = self.get_sage_neighbor_sampler(seeds=seeds)
            bz = sum(self.batch_size) if not self.seed_epoch else self.batch_size
            dl = DataLoader(
                seeds_encoded, batch_size=bz, shuffle=self.shuffle)
            
            if self.cache_result:
                self.cache = []
            for encoded_seeds in dl:
                batch_size, encoded_node_ids, adjs = sampler.sample(encoded_seeds)
                encoded_node_ids = encoded_node_ids.cpu().numpy()
                edge_ids = self.convert_sage_adjs_to_edge_ids(adjs)
                encoded_seeds = encoded_seeds.numpy()
                if self.cache_result:
                    self.cache.append([encoded_seeds, encoded_node_ids, edge_ids])
                yield encoded_seeds, encoded_node_ids, edge_ids
    
    def convert_sage_adjs_to_edge_ids(self, adjs):
        from torch_geometric.data.sampler import Adj
        if isinstance(adjs, Adj):
            adjs = [adjs]

        if '-merged' not in self.method:
            return [a[1].cpu().numpy() for a in adjs]
        
    def get_sage_neighbor_sampler(self, seeds):
        from torch_geometric.data.sampler import NeighborSampler
        g = self.g
        g.node_type_encode

        edge_index = g.edge_list_encoded
        edge_index = torch.LongTensor(edge_index)
        
        node_idx = np.array([g.node_encode[e] for e in seeds
                             if g.node_ts[e]< self.ts_range])
        
        node_idx = torch.LongTensor(node_idx)

        if self.method in {'sage', 'sage-merged'}:
            return NeighborSampler(
                sizes=self.width,
                edge_index=edge_index, 
                node_idx=node_idx, num_nodes=len(g.node_type),
                batch_size=sum(self.batch_size) if not self.seed_epoch else self.batch_size,
                num_workers=self.num_workers, 
                shuffle=self.shuffle
            )