from sklearn.cluster import KMeans
from .constructor import get_placeholder, get_model, get_optimizer, update
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
import pickle as pkl


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def train(X, A, n_clusters):
    n_iteration = 50
    model_str = 'arga_ae'
    adj = A

    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

    feas = {}
    feas['adj'] = adj
    features = sparse_to_tuple(sp.identity(X.shape[0]).tocoo())
    feas['features'] = features
    feas['num_features'] = features[2][1]
    feas['num_nodes'] = X.shape[0]

    feas['adj_norm'] = preprocess_graph(adj)
    feas['adj_label'] = adj_label
    feas['features_nonzero'] = features[1].shape[0]
    feas['pos_weight'] = pos_weight
    feas['norm'] = norm

    # Define placeholders
    placeholders = get_placeholder(feas['adj'])

    # construct model
    d_real, discriminator, ae_model = get_model(
        model_str,
        placeholders,
        feas['num_features'],
        feas['num_nodes'],
        feas['features_nonzero']
    )

    # Optimizer
    opt = get_optimizer(
        model_str,
        ae_model,
        discriminator,
        placeholders,
        feas['pos_weight'],
        feas['norm'],
        d_real,
        feas['num_nodes']
    )

    # Initialize session
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    # Train model
    embedding = None
    for epoch in range(n_iteration):
        embedding, _ = update(
            ae_model,
            opt,
            sess,
            feas['adj_norm'],
            feas['adj_label'],
            feas['features'],
            placeholders,
            feas['adj']
        )

        # if (epoch+1) % 2 == 0:
        #     predict_labels = KMeans(n_clusters=self.n_clusters, random_state=0).fit_predict(emb)
        #     print("Epoch:", '%04d' % (epoch + 1))
        #     cm = clustering_metrics(feas['true_labels'], predict_labels)
        #     cm.evaluationClusterModelFromLabel()
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embedding)

    return embedding, pred