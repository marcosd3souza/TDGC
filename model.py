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
        torch.manual_seed(random.randint(1, 1000))
        self.conv1 = GCNConv(n_input, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GCNConv(hidden_channels[2], n_output)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv4(x, edge_index)
        out = F.normalize(x)
        return out


class Model:
    def __init__(self, data, y_fake, y_true, n_clusters):
        self.n_hidden = [500, 300, 100]
        self.data = data
        self.y_fake = y_fake
        self.y_true = y_true
        self.n_clusters = n_clusters
        output = len(y_fake.unique())
        # self.A = torch.tensor(A, dtype=torch.float) #self.get_factored_normalized_adj(A)
        # D = pairwise_distances(data.x, metric='cosine')
        # self.D = torch.tensor(D, dtype=torch.float)
        self.gcn = GCN(n_input=self.data.x.shape[1], n_output=output,  hidden_channels=self.n_hidden)
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def kl_loss_function(self, out, target):
        z = F.normalize(out, p=2, dim=1)
        pred = z.argmax(dim=1)

        pred_probs = (np.bincount(pred) / pred.shape[0])
        pred_probs = torch.tensor(pred_probs, dtype=torch.float, requires_grad=True)

        target_probs = (np.bincount(target) / target.shape[0])
        target_probs = torch.tensor(target_probs, dtype=torch.float)

        return F.kl_div(pred_probs, target_probs, reduction='batchmean')

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_Q(self, z):
        # cluster layer
        # cluster_layer = Parameter(torch.Tensor(z.shape[0], z.shape[1]))
        # torch.nn.init.xavier_normal_(cluster_layer.data)
        # probs = (self.y_fake.bincount().reshape(1, -1) / self.y_fake.shape[0])
        # z_probs = (z.argmax(dim=1).bincount() / z.shape[0])

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.target_dist, 2), 1) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def _optimize(self, loss_fn1): # , loss_fn2):
        self.gcn.train()
        self.optimizer.zero_grad()

        out = self.gcn(self.data.x, self.data.edge_index)
        # batch_label = torch.tensor(np.random.normal(0, 1, size=(self.L.shape[0], self.data.num_classes)), dtype=torch.float)
        # loss = self.loss_function(adj_preds=out, adj_labels=batch_label)
        # adj_pred = torch.sigmoid(torch.matmul(out, out.t())).view(-1)

        # truth = torch.tensor(self.y_true, dtype=torch.long)
        # z = F.normalize(out, p=2, dim=1)
        # q = self.get_Q(z)
        # p = self.target_distribution(q.detach())
        #
        # kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        loss = self.kl_loss_function(out, self.y_fake)
        # loss_1 = loss_fn1(out, self.data.y)
        # loss_2 = loss_fn2(out, self.data.y)
        # A_pred = kneighbors_graph(out.detach().numpy(), n_neighbors=5).toarray()
        # A_pred = torch.tensor(A_pred, dtype=torch.float)
        # loss_2 = F.binary_cross_entropy(A_pred, self.A)
        # A_pred = torch.tensor(torch.matmul(out, out.t()), dtype=torch.float, requires_grad=True)
        # loss_2 = F.mse_loss(A_pred, self.A)
        # loss = kl_loss + loss_1
        # loss = self.criterion(out, self.data.y)
        # loss = self.criterion(out, torch.tensor(self.L.norm(dim=1), dtype=torch.long))
        # loss = self.criterion(out, torch.tensor(out.norm(dim=1), dtype=torch.long))
        loss.backward()
        self.optimizer.step()

        return loss

    def get_factored_normalized_adj(self, A):
        # D = pairwise_distances(A, metric='cosine')
        # D = pairwise_distances(X)
        A_norm = normalize(A, norm="l2")

        # nmf = NMF(
        #     n_components=10,
        #     init='random',
        #     max_iter=300
        # )
        #
        # W = nmf.fit_transform(A_norm)
        # H = nmf.components_
        #
        # D = W.dot(H)
        # A = kneighbors_graph(A_norm, n_neighbors=5, mode='connectivity').toarray()
        L = sparse.csgraph.laplacian(csgraph=A_norm, normed=True)
        # L = torch.tensor(L, dtype=torch.float)

        # labels = SpectralClustering(
        #     n_clusters=self.n_clusters,
        #     affinity='precomputed',
        #     random_state=np.random.randint(10000)
        # ).fit_predict(A_norm)

        # self.data.y = torch.tensor(labels, dtype=torch.long)

        return L

    def train(self, n_epochs=100):
        for epoch in tqdm(range(1, n_epochs + 1)):
            loss = self._optimize(self.criterion) # self.kl_loss_function, F.binary_cross_entropy)

            if epoch % 10 == 0:
                tqdm.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        self.gcn.eval()
        embedding = self.gcn(self.data.x, self.data.edge_index)
        # pred = embedding.argmax(dim=1).numpy()
        embedding = embedding.detach().numpy()
        # Z = TSNE(n_components=2).fit_transform(embedding)
        Z = umap.UMAP(
            n_neighbors=10,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(embedding)

        plt.scatter(Z[:, 0], Z[:, 1], c=self.y_true, cmap='Spectral')
        plt.show()

        # f_adj = np.matmul(embedding, np.transpose(embedding))
        # A = kneighbors_graph(embedding, n_neighbors=5).toarray()
        # A = self.get_factored_normalized_adj(A)
        # pred = SpectralClustering(self.n_clusters, affinity='precomputed').fit_predict(f_adj)
        pred = KMeans(self.n_clusters).fit_predict(embedding)
        # self.data.y = torch.tensor(pred, dtype=torch.long)
        #
        # self.gcn = GCN(n_input=self.data.x.shape[1], n_output=self.n_clusters, hidden_channels=self.n_hidden)
        #
        # for epoch in tqdm(range(1, n_epochs + 1)):
        #     loss = self._optimize(self.kl_loss_function)
        #     if epoch % 100 == 0:
        #         tqdm.write(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        #
        # # pred = self.get_spectral_pred(embedding)
        # self.gcn.eval()
        # embedding = self.gcn(self.data.x, self.data.edge_index)
        # pred = embedding.argmax(dim=1).numpy()
        # embedding = embedding.detach().numpy()

        return embedding, pred
