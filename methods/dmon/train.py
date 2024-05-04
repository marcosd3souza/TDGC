# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Graph Clustering with Graph Neural Networks.

===============================
This is the implementation of our paper,
[Graph Clustering with Graph Neural Networks]
(https://arxiv.org/abs/2006.16904).

The included code creates a DMoN (Deep Modularity Network) as introduced in the
paper.

Example execution to reproduce the results from the paper.
------
# From google-research/
python3 -m graph_embedding.dmon.train \
--graph_path=graph_embedding/dmon/data/cora.npz --dropout_rate=0.5
"""
# from typing import Tuple
# from absl import app
# from absl import flags
import numpy as np
import scipy.sparse
# from scipy.sparse import base
from sklearn import metrics
import tensorflow as tf#.compat.v2 as tf
from .dmon import DMoN
from .gcn import GCN
# from graph_embedding.dmon import metrics
from. import utils
tf.compat.v1.enable_v2_behavior()

ARCHITECTURE = [64, 48, 32]
N_EPOCHS=100
# N_CLUSTERS=15 # Cora
LEARNING_RATE = 0.001


def load_npz(
    filename
):
  """Loads an attributed graph with sparse features from a specified Numpy file.

  Args:
    filename: A valid file name of a numpy file containing the input data.

  Returns:
    A tuple (graph, features, labels, label_indices) with the sparse adjacency
    matrix of a graph, sparse feature matrix, dense label array, and dense label
    index array (indices of nodes that have the labels in the label array).
  """
  with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)
    adjacency = scipy.sparse.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    features = scipy.sparse.csr_matrix(
        (loader['feature_data'], loader['feature_indices'],
         loader['feature_indptr']),
        shape=loader['feature_shape'])

    label_indices = loader['label_indices']
    labels = loader['labels']
  assert adjacency.shape[0] == features.shape[
      0], 'Adjacency and feature size must be equal!'
  assert labels.shape[0] == label_indices.shape[
      0], 'Labels and label_indices size must be equal!'
  return adjacency, features, labels, label_indices


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


def build_dmon(input_features,
               input_graph,
               input_adjacency,
               n_clusters):
  """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

  Args:
    input_features: A dense [n, d] Keras input for the node features.
    input_graph: A sparse [n, n] Keras input for the normalized graph.
    input_adjacency: A sparse [n, n] Keras input for the graph adjacency.

  Returns:
    Built Keras DMoN model.
  """
  output = input_features
  for n_channels in ARCHITECTURE:
    output = GCN(n_channels)([output, input_graph])
  pool, pool_assignment = DMoN(n_clusters)([output, input_adjacency])
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def main(X, A, n_clusters):

  # Load and process the data (convert node features to dense, normalize the
  # graph, convert it to Tensorflow sparse tensor.
  # data_clean, data_dirty, labels = circular_gaussians(n_nodes, n_clusters)
  # adjacency, features, labels, label_indices = load_npz('data/cora.npz')
  features = scipy.sparse.csr_matrix(X)
  adjacency = A
  features = features.todense()
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(
      utils.normalize_graph(adjacency.copy()))

  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency, n_clusters)

  # Computes the gradients wrt. the sum of losses, returns a list of them.
  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
  model.compile(optimizer, None)

  for epoch in range(N_EPOCHS):
    loss_values, grads = grad(model, [features, graph_normalized, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))

  # Obtain the cluster assignments.
  _, assignments = model([features, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  pred = assignments.argmax(axis=1)  # Convert soft to hard clusters.

  return adjacency.toarray(), pred

  # Prints some metrics used in the paper.
  # print('Conductance:', metrics.conductance(adjacency, clusters))
  # print('Modularity:', metrics.modularity(adjacency, clusters))
  # nmi = metrics.normalized_mutual_info_score(labels, clusters[label_indices], average_method='arithmetic')
  # sil = metrics.silhouette_score(adjacency, clusters, metric='euclidean')
  # print('NMI: ', nmi)
  # print('SIL: ', sil)


# if __name__ == '__main__':
#   # app.run(main())
#     main()
