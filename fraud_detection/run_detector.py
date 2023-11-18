import os
import logging
import tempfile
import subprocess
from functools import partial
import glob

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

import tqdm
import joblib
import numpy as np
import pandas as pd
import datetime

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.utils import convert_tensor
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from xfraud.glib.fstore import FeatureStore
from xfraud.glib.brisk_utils import create_naive_het_graph_from_edges as _create_naive_het_graph_from_edges
from xfraud.glib.graph_loader import NaiveHetDataLoader, NaiveHetGraph
from xfraud.glib.pyg.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from xfraud.glib.utils import timeit

mem = joblib.Memory('./data/cache')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

create_naive_het_graph_from_edges = mem.cache(_create_naive_het_graph_from_edges)


def main(path_g, path_feat_db='data/store.db', path_result='exp_result.csv',
         dir_model='./model',
         conv_name='gcn', sample_method='sage',
         batch_size=(64, 16),
         width=16, depth=6,
         n_hid=400, n_heads=8, n_layers=6, dropout=0.2,
         optimizer='adamw', clip=0.25,
         n_batch=32, max_epochs=10, patience=8,
         seed_epoch=False, num_workers=0,
         seed=2020, debug=False, continue_training=False):
    """
    :param path_g:          path of graph file
    :param path_feat_db:    path of feature store db
    :param path_result:     path of output result csv file
    :param dir_model:       path of model saving
    :param conv_name:       model convolution layer type, choices ['', 'logi', 'gcn', 'gat', 'hgt', 'het-emb']
    :param sample_method:
    :param batch_size:      positive/negative samples per batch
    :param width:           sample width
    :param depth:           sample depth
    :param n_hid:           num of hidden state
    :param n_heads:
    :param n_layers:        num of convolution layers
    :param dropout:
    :param optimizer:
    :param clip:
    :param n_batch:
    :param max_epochs:
    :param patience:
    :param seed_epoch:      True -> iter on all seeds; False -> sample seed according to batch_size
    :param num_workers:
    :param seed:            random seed
    :param debug:           debug mode
    :param continue_training:
    :return:
    """
    stats = dict(
        batch_size=batch_size,
        width=width, depth=depth,
        n_hid=n_hid, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        conv_name=conv_name, optimizer=str(optimizer), clip=clip,
        max_epochs=max_epochs, patience=patience,
        seed=seed, path_g=path_g,
        sample_method=sample_method, path_feat_db=path_feat_db,
    )
    logger.info('Param %s', stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_feat_db_temp = f'{tmpdir}/store.db'


        with timeit(logger, 'fstore-init'):
            subprocess.check_call(
                f'cp -r {path_feat_db} {path_feat_db_temp}',
                shell=True)

            store = FeatureStore(path_feat_db_temp)
            
        if not os.path.isdir(dir_model):
            os.makedirs(dir_model)
        with timeit(logger, 'edge-load'):
            df_edges = pd.read_parquet(path_g)
        if debug:
            logger.info('Main in debug mode.')
            df_edges = df_edges.iloc[:10000]
        if 'seed' not in df_edges:
            df_edges['seed'] = 1
        with timeit(logger, 'g-init'):
            g = create_naive_het_graph_from_edges(df_edges)