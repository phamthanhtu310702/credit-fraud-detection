import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

import numpy as np
import tqdm
from xfraud.glib.graph_loader import GraphData, NaiveHetGraph
from xfraud.glib.utils import timeit


def create_naive_het_graph_from_edges(df):
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['src', 'ts']].drop_duplicates()
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        view = df[['src', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        view = df[['dst', 'dst_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))

    if 'graph_edge_type' not in df:
        df['graph_edge_type'] = 'default'

    with timeit(logger, 'edge-list-init'):
        edge_list = list(
            df[['src', 'dst', 'graph_edge_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]

    select = df['seed'] > 0
    view = df[select][['src', 'src_label']].drop_duplicates()
    seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    return NaiveHetGraph(node_type, edge_list,
                         seed_label=seed_label, node_ts=node_ts)
