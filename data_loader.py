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
from torch_geometric.datasets import Planetoid, DBLP, CoraFull
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from enum import Enum

# from gtda.homology import VietorisRipsPersistence
from tmap.tda import mapper, Filter
from tmap.tda.cover import Cover
from sklearn.cluster import DBSCAN

# from graph import DataGraph, convert_to_graph


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
    CORA = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    CORAFULL = Planetoid(root='data/CitationFull', name='Cora', transform=NormalizeFeatures())
    CITESEER = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
    PUBMED = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

    # Heterogeneous Dataset
    DBLP = DBLP(root='data/DBLP')

    # non-graph data
    # CIFAR10 = NonGraphData().load_cifar10()
    # USPS = NonGraphData(path='./data/USPS.mat').load_usps()
    # MNIST = NonGraphData(path='./data/MNIST_2k2k.mat').load_mnist()


class DataLoader:
    def __init__(self, dataset: Dataset):
        self.data = dataset.value
        self.n_clusters = dataset.value.num_classes

    def load(self):
        self.X = self.data.x.numpy()
        self.y = self.data.y.numpy()
        self.edge_index = self.data.edge_index

        return self.construct_data()

    def construct_data(self):
        data = Data(
            x=torch.tensor(self.X, dtype=torch.float),
            y=torch.tensor(self.y, dtype=torch.long),
            edge_index=self.edge_index,
            num_classes=self.n_clusters
        )

        print(f'Dataset: {data}:')
        print('======================')
        print(f'Number of graphs: {len(data)}')
        print(f'Number of features: {data.num_features}')
        print(f'Number of classes: {data.num_classes}')

        return data
