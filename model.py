import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import random

from sklearn.metrics import pairwise_distances
from tqdm import tqdm


# from sklearn import metrics
# import utils


class GCN(torch.nn.Module):
    def __init__(self, n_input, n_output, hidden_channels):
        super().__init__()
        # torch.manual_seed(random.randint(1, 1000))
        self.conv1 = GCNConv(n_input, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        # self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv3 = GCNConv(hidden_channels[1], n_output)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)
        # out = F.normalize(x)
        return x


class Model:
    def __init__(self, data_pos, data_neg, n_clusters):
        
        self.n_hidden = [100, 30]
        self.data_pos = data_pos
        self.data_neg = data_neg

        n_features = self.data_pos.x.shape[1]
        self.gcn = GCN(n_input=n_features, n_output=n_clusters,  hidden_channels=self.n_hidden)
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=0.001, weight_decay=0.0001)#5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def contrastive_priori_loss(self, z1, z2, y_priori, margin=1.5):
        #priori
        priori_loss = self.criterion(z1, y_priori)

        # Normalize the embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Compute the positive and negative pairs
        positive_pairs = torch.sum((z1 - z2) ** 2, dim=1)
        negative_pairs = torch.sum((z1 - (-z2)) ** 2, dim=1)

        # Contrastive loss
        contrastive_loss = torch.mean(torch.relu(positive_pairs - margin) + negative_pairs)
        return contrastive_loss + priori_loss


    def _optimize(self): # , loss_fn2):
        self.gcn.train()
        self.optimizer.zero_grad()

        z_pos = self.gcn(self.data_pos.x, self.data_pos.edge_index)
        z_neg = self.gcn(self.data_neg.x, self.data_neg.edge_index)   

        loss = self.contrastive_priori_loss(z_pos, z_neg, self.data_pos.y)
        
        loss.backward()
        self.optimizer.step()

        return loss.item()  

    def train(self, n_epochs=150):
        for epoch in tqdm(range(1, n_epochs + 1)):
            loss = self._optimize()

            if epoch % 10 == 0:
                tqdm.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        self.gcn.eval()               
        with torch.no_grad():
            embedding = self.gcn(self.data_pos.x, self.data_pos.edge_index)
            
        pred = np.argmax(embedding, axis=1)

        return embedding, pred
