import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from hnswlib import Index

def TGM(data, m):
    K = m.shape[0]
    d = cdist(m, data)
    min_idx = np.argmin(d, axis=0)
    pred = min_idx
    pairwise_sep = one_dim_gm_pdf(data, m, pred)
    pred_all, global_sep, global_com, cluster_num = global_measures(data, pairwise_sep, pred, K)
    global_sep = global_sep / np.max(global_sep)
    global_com = global_com / np.max(global_com)
    sep_com = global_sep + global_com
    return pred_all, cluster_num, global_sep, global_com, sep_com


def one_dim_gm_pdf(data, m, pred):
    K = m.shape[0]
    pairwise_sep = np.zeros((K, K))

    for i in range(K):
        for j in range(i + 1, K):
            c_i = np.mean(data[pred == i, :], axis=0)
            c_j = np.mean(data[pred == j, :], axis=0)
            c_0 = (c_i + c_j) / 2

            data_i = data[pred == i, :]
            data_j = data[pred == j, :]
            data_i_proj = (data_i - c_0) @ (c_i - c_j) / np.linalg.norm(c_i - c_j) ** 2
            data_j_proj = (data_j - c_0) @ (c_i - c_j) / np.linalg.norm(c_i - c_j) ** 2
            data_i_proj = data_i_proj.reshape(-1, 1)  # 将数据转换为列向量
            data_j_proj = data_j_proj.reshape(-1, 1)  # 将数据转换为列向量

            sigma_mat = np.zeros((2, 1, 1))
            sigma_mat[0, :, :] = np.std(data_j_proj) ** 2
            sigma_mat[1, :, :] = np.std(data_i_proj) ** 2

            GMModel = {}
            GMModel['mu'] = np.array([[0.5], [-0.5]])
            weights = np.array([np.sum(pred == i), np.sum(pred == j)]) / (np.sum(pred == i) + np.sum(pred == j))
            y = multivariate_normal.pdf(np.arange(-0.5, 0.51, 0.01), mean=GMModel['mu'][0], cov=sigma_mat[0]) * weights[0] + \
                multivariate_normal.pdf(np.arange(-0.5, 0.51, 0.01), mean=GMModel['mu'][1], cov=sigma_mat[1]) * weights[1]
            pairwise_sep[i, j] = 1 / np.min(y)

    pairwise_sep = np.maximum(pairwise_sep, pairwise_sep.T)
    return pairwise_sep

def global_measures(data, pairwise_sep, pred, K):
    global_com = np.zeros(K-1)
    group_set = [[i] for i in range(K)]
    pred_all = np.empty((K-1, len(pred)), dtype=int)  # 预定义一个空数组
    for k in range(K, 1, -1):
        s = np.inf * np.ones((k, k))
        for i in range(1, k):
            for j in range(i):
                ss = []
                for ii in group_set[i]:
                    for jj in group_set[j]:
                        ss.append(pairwise_sep[ii, jj])
                s[i, j] = np.min(ss)

        min_s = np.min(s)
        min_idx = np.argmin(s)
        i, j = np.unravel_index(min_idx, (k, k))

        global_com[k-2] = min_s
        group_set.append(group_set[i] + group_set[j])
        group_set.pop(i)
        group_set.pop(j)

        new_pred = np.zeros_like(pred)
        for idx, group in enumerate(group_set):
            new_pred[np.isin(pred, group)] = idx + 1
        pred_all[k-2, :] = new_pred

    cluster_num = [len(np.unique(pred_all[i, :])) for i in range(pred_all.shape[0])]
    ia = np.unique(cluster_num, return_index=True)[1]
    k = 10
    nn_idx = fast_ann(data,k)
    global_sep = []
    for i in ia:
        global_sep.append(calculate_global_sep(pred_all[i, :], nn_idx))
        global_sep[i] = np.max(global_sep[:i + 1])
    return pred_all, global_sep, global_com, cluster_num


def calculate_global_sep(pred, nn_idx):
    u_label = np.unique(pred)
    cluster_num = len(u_label)
    sep = np.zeros(cluster_num)
    for i in range(cluster_num):
        sep[i] = np.sum(np.isin(nn_idx[:, pred == u_label[i]], np.where(pred != u_label[i]))) / nn_idx.shape[0]
    sep = np.max(sep)
    return sep


def fast_ann(data,k):
    index = Index(space='l2', dim=data.shape[1])
    index.init_index(max_elements=len(data), ef_construction=100, M=16)
    index.add_items(data)
    index.set_ef(10)
    nn_idx = np.zeros((k, data.shape[0]), dtype=int)
    for i in range(data.shape[0]):
        labels,_ = index.knn_query(data[i], k+1)
        nn_idx[:, i] = labels[:, 1:]
    return nn_idx



