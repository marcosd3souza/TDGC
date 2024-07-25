import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

import utils
import numpy as np
import pandas as pd
from scipy import sparse

from data_loader import DataLoader, Dataset

from model import Model
from graph_data import TopologicalGraph

## SOTA Methods
# from methods.SCGC import scgc
# from methods.ARGA import arga
# from methods.DFCN import dfcn_run
# from methods.DAEGC import daegc
from methods.dmon import train as dmon
# from methods.CCGC import ccgc
# from methods.AGE import age


def get_row_df(method, name, sil, ch, db, acc, nmi, ari):
    row = {
        'method': [method],
        'dataset': [name],
        # 'embedding': [embedding],
        'sil': [sil],
        'ch': [ch],
        'db': [db],
        'acc': [acc],
        'nmi': [nmi],
        'ari': [ari]
    }

    return row


results = pd.DataFrame()
datasets = [
    Dataset.CORA,
    Dataset.CORAFULL,
    Dataset.CITESEER,
    # Dataset.PUBMED, # Very large graph
    # Dataset.DBLP # heterogeneous data
    # Dataset.CIFAR10, # muito difícil (DMoN não converge)
    # Dataset.MNIST,
    # Dataset.USPS
]

for dataset in datasets:
    print(f'------------------- data: {dataset.name}')

    data = DataLoader(dataset).load()
    
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index
    n_clusters = data.num_classes
    y_true = data.y
    y_true_sorted = y_true[np.argsort(y_true)]

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    A[edge_index[0], edge_index[1]] = 1.0
    A = sparse.csc_matrix(A.numpy())

    for it in range(10):

        # print('----------------------- Baseline Kmeans')
        # y_kmeans = KMeans(n_clusters).fit_predict(data.x.numpy())
        # y_kmeans = torch.tensor(y_kmeans, dtype=torch.long)
        # sil, ch, db, acc, nmi, ari = utils.evaluate(data.x, y_true, y_kmeans)
        # row = get_row_df('baseline_kmeans', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        # print('---------------------------------')
        #
        # print('----------------------- Baseline Louvain')
        # sil, ch, db, acc, nmi, ari = utils.evaluate(data.x, y_true, y_louvain)
        # row = get_row_df('baseline_louvain', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        # print('---------------------------------')        

        # OURS
        # print('----------------------- OURS (KMeans)')
        # embedding, y_pred = Model(data, y_kmeans, A_arr, y_true, n_clusters).train()
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('GCN_KMeans', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)

        print('----------------------- OURS (TDGC)')
        data_pos, data_neg = TopologicalGraph(data.x, n_clusters).get_contrastive_data_graph()
        embedding, y_pred = Model(data_pos, data_neg, n_clusters).train()
        # results
        sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true_sorted, y_pred)
        row = get_row_df('TDGC', dataset.name, sil, ch, db, acc, nmi, ari)
        results = pd.concat([results, pd.DataFrame(row)], axis=0)
        print('---------------------------------')


        # DMoN
        # print('----------------------- DMoN')
        # embedding, y_pred = dmon.main(data.x, A, n_clusters)
        # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('DMoN', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        # print('---------------------------------')
        #
        # # SCGC
        # print('----------------------- SCGC')
        # embedding, y_pred = scgc.train(data.x, A, cluster_num=n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('SCGC', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)

        # ARGA
        # embedding, y_pred = arga.train(data.x, A, n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('ARGA', it, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)

        # # DFCN
        # print('----------------------- DFCN')
        # embedding, y_pred = dfcn_run.run(data.x, A, n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('DFCN', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        #
        # # DAEGC *(best)
        # print('----------------------- DAEGC')
        # embedding, y_pred = daegc.run(data, n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('DAEGC', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        #
        # # CCGC
        # print('----------------------- CCGC')
        # embedding, y_pred = ccgc.train(data.x, A, n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('CCGC', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)
        #
        # AGE
        # print('----------------------- AGE')
        # embedding, y_pred = age.gae_for(data.x, A, n_clusters)
        # # results
        # sil, ch, db, acc, nmi, ari = utils.evaluate(embedding, y_true, y_pred)
        # row = get_row_df('AGE', dataset.name, sil, ch, db, acc, nmi, ari)
        # results = pd.concat([results, pd.DataFrame(row)], axis=0)

results.to_csv('benchmark_07_2024.csv', sep=';', index=False)
# print('------------------------------------------ RESULTS -------------------------------------------')
# methods = [
#     'baseline_kmeans',
#     'baseline_louvain',
#     'GCN_KMeans',
#     'GCN_Louvain',
#     # 'DMoN',
#     # 'SCGC',
#     # 'DFCN',
#     # 'DAEGC',
#     # 'CCGC',
#     # 'AGE'
# ]
# for method in methods:
#     print(f'----------- {method}')
#     print(results[results['method'] == method].iloc[:, 2:].describe())
