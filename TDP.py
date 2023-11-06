from scipy.spatial.distance import euclidean
import numpy as np


def cal(a,topology,data):
    num_data = data.shape[0]
    adjmatrix = np.zeros((num_data, num_data), dtype=int)  # matrix是邻接矩阵
    for edge in topology:
        vertex1, vertex2 = edge
        adjmatrix[vertex1, vertex2] = 1
        adjmatrix[vertex2, vertex1] = 1
    distance_matrix = np.zeros((num_data, num_data))
    # 计算节点之间的距离
    for i in range(num_data):
        for j in range(num_data):
            if adjmatrix[i, j] == 1:
                # 使用欧式距离计算节点之间的距离
                distance_matrix[i, j] = euclidean(data[i], data[j])
            else:
                distance_matrix[i, j] = 0
    temp = distance_matrix[a].tolist()
    indices_down = [value for index, value in enumerate(temp) if value != 0]
    dka = sum(indices_down)/len(indices_down)
    dists_up = []
    temp = adjmatrix[a].tolist()
    indices_down = [index for index, value in enumerate(temp) if value == 1]
    for i in indices_down:
        temp = distance_matrix[i].tolist()
        indices_up = [value for index, value in enumerate(temp) if value != 0]
        dists_up.append(sum(indices_up)/len(indices_up))
    return sum(dists_up)/(len(dists_up)*dka)



def TDP(data, topology):
    output = []
    for i in range(np.shape(data)[0]):
        if cal(i,topology,data) > 1:
                output.append(i)

    output = list(set(output))
    return data[output],output












