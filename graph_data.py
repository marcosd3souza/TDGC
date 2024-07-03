import umap
from torch_geometric.data import Data

from tmap.tda.metric import Metric
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover

from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

# from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from networkx import Graph
import networkx as nx


class TopologicalGraph:
    def __init__(self, X=None, n_clusters=None):
        self.X = X
        self.K = n_clusters

        self.D = squareform(pdist(X, metric="cosine"))
        self.D[self.D / 1 != self.D] = 1

        self.mapper = mapper.Mapper(verbose=1)

    def _spectral_clustering(self, adj_matrix):
        spectral = SpectralClustering(n_clusters=self.K, affinity='precomputed', assign_labels='kmeans')
        labels = spectral.fit_predict(adj_matrix.toarray())

        # # Compute the degree matrix
        # degrees = np.sum(adj_matrix, axis=1)
        # D = np.diag(degrees)

        # # Compute the Laplacian matrix
        # L = D - adj_matrix

        # L = L.astype(float)

        # # Compute the eigenvalues and eigenvectors of the Laplacian
        # eigvals, eigvecs = eigsh(L, k=self.K, which='SM')

        # # Perform k-means clustering on the eigenvectors
        # kmeans = KMeans(n_clusters=self.K)
        # kmeans.fit(eigvecs)
        # labels = kmeans.labels_

        return labels

    def projection(self, representation):
        lens = [Filter.UMAP(metric=Metric('precomputed'))]
        return self.mapper.filter(representation, lens=lens)

    def _get_tda_adj(self):
        projection = self.projection(self.D)
        cover = Cover(projected_data=projection, resolution=20, overlap=0.75)

        graph = self.mapper.map(data=projection, cover=cover)
        A_tda = nx.adjacency_matrix(graph)#.toarray()

        return A_tda, graph               

    def _get_adj_labels(self):

        A_tda, graph = self._get_tda_adj()
        y_tda = self._spectral_clustering(A_tda)

        # map tda labels to X labels
        labels = np.array([-1] * self.X.shape[0])
        for node_id in graph.nodes:
            label = int(y_tda[node_id])
            labels[graph.nodes[node_id]['sample']] = label
                
        # sort X and labels
        sort_idx = np.argsort(labels)
        labels = labels[sort_idx]        
        self.X = self.X[sort_idx]                
        
        # Number of nodes
        num_nodes = len(labels)

        # Initialize the adjacency matrix with zeros
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Fill the adjacency matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if labels[i] == labels[j]:
                    adjacency_matrix[i][j] = 1

        return adjacency_matrix, labels

    def get_contrastive_data_graph(self):
        A, labels = self._get_adj_labels()

        # tensors
        G_A = nx.from_numpy_array(A)
        edge_index_A = torch.tensor(list(G_A.edges), dtype=torch.long).t().contiguous()
        
        A = torch.tensor(A, dtype=torch.float)
        graph_y = torch.tensor(labels, dtype=torch.long)
        x = torch.tensor(self.X, dtype=torch.float)

        # random adj (negative examples)
        B = np.random.randint(2, size=(A.shape[0], A.shape[0]))
        G_B = nx.from_numpy_array(B)
        edge_index_B = torch.tensor(list(G_B.edges), dtype=torch.long).t().contiguous()

        data_pos = Data(x=x, edge_index=edge_index_A, y=graph_y)
        data_neg = Data(x=x, edge_index=edge_index_B)

        return data_pos, data_neg        
    
