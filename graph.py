import umap
from networkx import Graph
from sklearn.decomposition import NMF
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import networkx as nx
import community as community_louvain
from community import generate_dendrogram
import numpy as np
import torch


def convert_to_graph(data):
    A = nx.adjacency_matrix(data).toarray()
    return nx.from_numpy_array(A)


class DataGraph:
    def __init__(self, X=None, n_clusters=None):
        self.X = X
        self.K = n_clusters

    def get_knn_graph(self):
        # D = pairwise_distances(self.X, metric='cosine')
        # nmf = NMF(
        #     n_components=10,
        #     init='random',
        #     max_iter=300
        # )
        #
        # W = nmf.fit_transform(D)
        # H = nmf.components_
        #
        # Dt = W.dot(H)
        # Z = umap.UMAP(
        #     n_neighbors=5,
        #     min_dist=0.0,
        #     n_components=10,
        #     random_state=42,
        # ).fit_transform(self.X)

        A = kneighbors_graph(self.X, n_neighbors=9, mode='connectivity', metric='cosine')#.toarray()
        G = nx.from_numpy_array(A)

        return G, A

    def get_graph_communities(self, G, res=None):
        # n_clusters = None
        if res is not None:
            y_fake, modularity = self._get_y_fake(res, G)
            y_fake = torch.tensor(y_fake, dtype=torch.long)
            n_clusters = len(y_fake.unique())
            print(f'best modularity: {modularity}')
            print(f'best K: {n_clusters}')
        else:
            res = np.linspace(0.1, 3.0, 20)
            # y_fake = None
            best_y_fake = None
            best_modularity = 0
            min_dist = 9999
            for r in res:
                print('resolution: ', r)

                y_fake, modularity = self._get_y_fake(r, G)
                best_K = len(np.unique(y_fake))
                if abs(best_K - self.K) < min_dist:
                    print(f'best modularity: {modularity}')
                    print(f'best K: {best_K}')
                    # best_modularity = modularity
                    best_y_fake = y_fake
                    min_dist = abs(best_K - self.K)

                    if min_dist == 0:
                        break
                # if len(np.unique(y_fake)) == self.K:
                #     n_clusters = self.K
                #     print('------------------------------ Found K !!!')
                #     break

            # if n_clusters is None:
            #     y_fake = self._get_y_fake(1.0, G)
            #     n_clusters = len(np.unique(y_fake))

            y_fake = torch.tensor(best_y_fake, dtype=torch.long)
            n_clusters = len(np.unique(y_fake))
        return y_fake, n_clusters

    @staticmethod
    def _get_y_fake(resol, G):
        partition = community_louvain.best_partition(G, resolution=resol, randomize=False)
        v = {}

        for key, value in partition.items():
            v.setdefault(value, set()).add(key)

        communities = [i for i in v.values()]

        y_fake = np.zeros(G.number_of_nodes()) - 1

        for cls_idx, items_idx in enumerate(communities):
            y_fake[list(items_idx)] = int(cls_idx)

        modularity = nx.community.modularity(G, communities)
        return y_fake, modularity
