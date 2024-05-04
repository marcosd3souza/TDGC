from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np


def evaluate(embedding, y_true, y_predict):
    cm = metrics.confusion_matrix(y_true, y_predict)
    _make_cost_m = lambda x: -x + np.max(x)
    indexes = linear_assignment(_make_cost_m(cm))
    indexes = np.concatenate([indexes[0][:, np.newaxis], indexes[1][:, np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)

    sil = metrics.silhouette_score(embedding, y_predict, metric='euclidean')
    ch = metrics.calinski_harabasz_score(embedding, y_predict)
    db = metrics.davies_bouldin_score(embedding, y_predict)

    nmi = metrics.normalized_mutual_info_score(y_true, y_predict)
    ari = metrics.adjusted_rand_score(y_true, y_predict)

    print('---------------------- clustering performance ------------------------')
    print('----------------- unsupervised')
    print(f'sil: {sil}')
    # print(f'ch: {ch}')
    # print(f'db: {db}')
    print('----------------- supervised')
    print(f'acc: {acc}')
    print(f'nmi: {nmi}')
    print(f'ari: {ari}')
    print('----------------------------------------------------------------------')

    return sil, ch, db, acc, nmi, ari
