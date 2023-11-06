from TGM import *
import numpy as np
from scipy.special import comb
from sklearn.metrics import normalized_mutual_info_score


def show_result(data, label, pred_all, cluster_num, measure_sep, measure_com, sep_com):
    plt.figure()
    plt.plot(cluster_num, sep_com, '-s', markersize=5)
    plt.plot(cluster_num, measure_sep, '-s', markersize=5)
    plt.plot(cluster_num, measure_com, '-s', markersize=5)
    plt.legend(['sep+com', 'sep', 'com'])
    plt.xlabel('Global measures')
    u_pred = np.unique(label)
    plt.figure()
    for i in range(len(u_pred)):
        plt.plot(data[label.flatten() == u_pred[i], 0], data[label.flatten() == u_pred[i], 1], '.', markersize=10)
    plt.xlabel('Ground truth')
    sep_com[0] = np.inf
    median_pos = np.argmin(sep_com)
    pred = pred_all[median_pos, :]
    pred = label_correction(label.flatten(), pred, 1)  # Define label_correction function accordingly
    u_pred = np.unique(pred)
    plt.figure()
    for i in range(len(u_pred)):
        plt.plot(data[pred == u_pred[i], 0], data[pred == u_pred[i], 1], '.', markersize=10)
    plt.xlabel('TSSC clustering result')
    plt.show()
    TSSC_result = clustering_evaluate(label.flatten().T, pred.T)  # Define clustering_evaluate function accordingly
    return TSSC_result

def label_correction(class_, label, mode):
    new_label = np.zeros(len(class_))
    uc = np.unique(class_)
    ul = np.unique(label)
    num_uc = len(uc)
    num_ul = len(ul)

    if mode == 1:
        # assign by ratio
        for i in range(num_ul):
            nij = np.zeros(num_uc)
            for j in range(num_uc):
                nij[j] = np.sum((label == ul[i]) & (class_ == uc[j])) / np.sum(class_ == uc[j])
            max_idx = np.argmax(nij)
            new_label[label == ul[i]] = uc[max_idx]
    else:
        # assign by order
        label_num = np.array([np.sum(label == i) for i in ul])
        sorted_indices = np.argsort(-label_num)  # 从大到小排序并获取索引

        # 构建索引字典
        index_dict = {}
        for idx, value in enumerate(label_num):
            if value not in index_dict:
                index_dict[value] = []
            index_dict[value].append(idx)

        # 按照数量从大到小的顺序构建输出索引列表
        sorted_indices_with_same_count = []
        for value, indices in sorted(index_dict.items(), reverse=True):
            sorted_indices_with_same_count.extend(indices)
        sort_idx = sorted_indices_with_same_count
        for i in range(num_ul):
            new_label[label == ul[sort_idx[i]]] = i + 1
    return new_label


def clustering_evaluate(target, result):
    result = label_correction(target, result, 2)
    target = label_correction(target, target, 2)
    data_num = len(target)
    target_length = len(np.unique(target))
    result_length = len(np.unique(result))

    b = np.zeros(target_length)
    cb = 0
    cd = 0
    for i in range(target_length):
        b[i] = np.sum(target == i+1)
        if b[i] >= 2:
            cb += comb(int(b[i]), 2)
        else:
            cd += 0

    d = np.zeros(result_length)
    for i in range(result_length):
        d[i] = np.sum(result == i+1)
        if d[i] >= 2:
            cd += comb(int(d[i]), 2)
        else:
            cd += 0

    n = np.zeros((target_length, result_length))
    fval = np.zeros((target_length, result_length))
    cn = 0
    for i in range(target_length):
        for j in range(result_length):
            n[i, j] = np.sum((target == i+1) & (result == j+1))
            if n[i, j] >= 2:
                cn += comb(int(n[i, j]), 2)
            else:
                cn += 0
            if b[i] > 0:
                rec = n[i, j] / b[i]
            else:
                rec = 0
            if d[j] > 0:
                pre = n[i, j] / d[j]
            else:
                pre = 0

            if rec == 0 and pre == 0:
                fval[i,j] = 0
            else:
                fval[i, j] = 2 * rec * pre / (rec + pre)

    n_max = np.max(n, axis=1)
    max_idx = np.argmax(n, axis=1)
    acc = np.sum(n_max) / data_num
    if np.sum(n_max) > 0:
        pre = np.sum(n_max / b) / target_length
    else:
        pre = 0
    if np.sum(n_max) > 0:
        rec = np.sum(n_max / d[max_idx]) / target_length
    else:
        rec = 0
    fval_sum = np.sum(b / data_num * np.max(fval, axis=1))

    temp = (cb * cd) / comb(data_num, 2)
    ari = (cn - temp) / (0.5 * (cb + cd) - temp)

    temp = 0
    for l in range(target_length):
        for h in range(result_length):
            if b[l] * d[h] > 0:
                temp = temp + 2 * n[l, h] / data_num * np.log(n[l, h] * data_num / (b[l] * d[h]) + np.finfo(float).eps)
            else:
                temp = 0
    nmi_value = normalized_mutual_info_score(target.flatten(), result.flatten())

    if result_length == 1:
        dcv = abs(np.sqrt(np.sum((b - np.mean(b))**2) / (target_length - 1)) / np.mean(b))
    else:
        dcv = abs(np.sqrt(np.sum((b - np.mean(b))**2) / (target_length - 1)) / np.mean(b) -
                  np.sqrt(np.sum((d - np.mean(d))**2) / (result_length - 1)) / np.mean(d))

    cluster_num = len(np.unique(result))

    output = {
        "acc": acc,
        "pre": pre,
        "rec": rec,
        "fval": fval_sum,
        "ari": ari,
        "nmi": nmi_value,
        "dcv": dcv,
        "cluster_num": cluster_num
    }

    return output

def nmi(x, y):
    # Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    n = len(x)
    x = x.reshape(1, n)
    y = y.reshape(1, n)

    l = min(min(x), min(y))
    x = x - l + 1
    y = y - l + 1
    k = max(max(x), max(y))

    idx = np.arange(n)
    Mx = np.zeros((n, k))
    My = np.zeros((n, k))
    Mx[idx, x - 1] = 1
    My[idx, y - 1] = 1
    Pxy = (Mx.T @ My) / n
    Pxy = Pxy[Pxy > 0]
    Hxy = -np.sum(Pxy * np.log2(Pxy))

    Px = np.mean(Mx, axis=0)
    Py = np.mean(My, axis=0)
    Px = Px[Px > 0]
    Py = Py[Py > 0]

    Hx = -np.sum(Px * np.log2(Px))
    Hy = -np.sum(Py * np.log2(Py))

    MI = Hx + Hy - Hxy
    z = np.sqrt((MI / Hx) * (MI / Hy))
    z = max(0, z)

    return z

