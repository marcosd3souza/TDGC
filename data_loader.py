import cv2
# import networkx as nx
import numpy as np
import torch
# from gtda.diagrams import PersistenceEntropy
from keras.datasets import cifar10
from scipy.sparse import lil_matrix
import scipy.io as sio
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tmap.tda.metric import Metric
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Planetoid, DBLP
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from enum import Enum

from gtda.homology import VietorisRipsPersistence
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from sklearn.cluster import DBSCAN

from graph import DataGraph, convert_to_graph


class NonGraphData:
    def __init__(self, path=None):
        self.path = path

    def load_cifar10(self):
        # CIFAR10
        (train_images, train_labels), _ = cifar10.load_data()
        labels = np.squeeze(train_labels)
        idx = np.array([np.random.choice(np.where(labels == l)[0], 100) for l in np.unique(labels)]).flatten()
        cifar10_df = train_images[idx]
        # grayscale
        cifar10_df = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifar10_df])
        cifar10_df = cifar10_df.reshape(1000, 1024)
        #
        cifar10_labels = labels[idx]

        x = torch.tensor(cifar10_df, dtype=torch.float)
        y_true = torch.tensor(cifar10_labels, dtype=torch.long)

        return Data(x=x, y=y_true, transform=NormalizeFeatures())

    def load_mnist(self):
        # MNIST
        data = sio.loadmat(self.path)  # './datasets/MNIST_2k2k.mat')
        mnist_df = data['fea']
        mnist_labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)

        x = torch.tensor(mnist_df, dtype=torch.float)
        y_true = torch.tensor(mnist_labels, dtype=torch.long)

        return Data(x=x, y=y_true, transform=NormalizeFeatures())

    def load_usps(self):
        # USPS
        data = sio.loadmat(self.path)  # './datasets/USPS.mat')
        labels = np.squeeze(data['gnd'] - data['gnd'].min() + 1)
        idx = np.array([
            np.where(labels == i)[0][0:100] for i in np.unique(labels) if len(np.where(labels == i)[0]) >= 100
        ]).flatten()
        usps_df = data['fea'][idx]
        usps_labels = labels[idx]
        x = torch.tensor(usps_df, dtype=torch.float)
        y_true = torch.tensor(usps_labels, dtype=torch.long)

        return Data(x=x, y=y_true, transform=NormalizeFeatures())


class Dataset(Enum):
    # Homogeneous Datasets
    CORA = Planetoid(root='data/Planetoid', name='Cora')
    CITESEER = Planetoid(root='data/Planetoid', name='CiteSeer')
    PUBMED = Planetoid(root='data/Planetoid', name='PubMed')

    # Heterogeneous Dataset
    DBLP = DBLP(root='data/DBLP')

    # non-graph data
    CIFAR10 = NonGraphData().load_cifar10()
    USPS = NonGraphData(path='./data/USPS.mat').load_usps()
    MNIST = NonGraphData(path='./data/MNIST_2k2k.mat').load_mnist()


class DataLoader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def construct_W(self, y):
        label = np.unique(y)
        n_samples = len(y)
        n_classes = np.unique(y).size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        W = lil_matrix((n_samples, n_samples))
        for i in range(n_classes):
            class_idx = (y == label[i])
            class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
            W[class_idx_all] = 1.0 # / np.sum(np.sum(class_idx))
        return W

    def load(self):
        self.X = self.dataset.value.x
        y_true = self.dataset.value.y.numpy()

        D = squareform(pdist(self.X.numpy(), metric="cosine"))

        ### TDA
        # Step1. initiate a Mapper
        tm = mapper.Mapper(verbose=1)
        # Step2. Projection
        lens = [Filter.UMAP(metric=Metric(metric="precomputed"))]
        self.projected_X = tm.filter(D, lens=lens)
        # Step3. Create Cover
        cover = Cover(projected_data=self.projected_X, resolution=20, overlap=0.75)
        # Step4. Create Graph
        G_tda = tm.map(data=self.projected_X, cover=cover)
        G_tda = convert_to_graph(G_tda)

        graph = DataGraph(self.projected_X, len(np.unique(y_true)))

        G, A = graph.get_knn_graph()
        y, n_clusters = graph.get_graph_communities(G_tda, res=1.0)
        # center_means = [np.log(np.mean(projected_X[y == i])) for i in range(n_clusters)]
        # target_distribution = self.get_target_distribution(center_means)
        # W = self.construct_W(y).toarray()
        # G = nx.from_numpy_array(W)
        edge_index = from_networkx(G).edge_index # self.dataset.value.edge_index

        data = self.construct_data(y, n_clusters, edge_index)

        return data, A, y, y_true, n_clusters

    def construct_data(self, y, n_clusters, edge_index):
        data = Data(
            x=torch.tensor(self.projected_X, dtype=torch.float),
            y=y,
            edge_index=edge_index,
            num_classes=n_clusters,
            transform=NormalizeFeatures()
        )

        print(f'Dataset: {data}:')
        print('======================')
        print(f'Number of graphs: {len(data)}')
        print(f'Number of features: {data.num_features}')
        print(f'Number of classes: {data.num_classes}')

        return data
