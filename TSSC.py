# -----------------------------------------------------------------------------
# code created by Yongqi Xu
# -----------------------------------------------------------------------------
import sys
import TR
import time
from sklearn.cluster import KMeans
from TDP import *
from showresult import *
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    start = time.time()
    dataname = 'lithuanian' # banana lithuanian ids2 gaussian
    seed       = 1
    topology   = "regular"
    neuron_num = 100
    n_neighbor = 5
    RSOM_epochs  = 25000
    kmeans_epoch = 5
    sigma      = 0.5, 0.01
    lrate      = 0.50, 0.05
    if seed is None:
        seed = np.random.randint(0,100)
    np.random.seed(seed)


    #retrieve input data
    columns = ['x', 'y']
    with open('./datasets/'+dataname+'.txt', 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    datas = np.array(data, dtype=float)
    N, D = np.shape(datas)
    scaler1 = MinMaxScaler()
    datas = scaler1.fit_transform(datas)
    X = datas
    Y = None


    #step1:TR
    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som = TR.SOM(neuron_num, topology, n_neighbor)
    print("done!")
    som.fit(X, None, RSOM_epochs, sigma=sigma, lrate=lrate)    #RSOM training
    neurons = som.codebook["X"].reshape(len(som.codebook), D)
    kmeans = KMeans(n_clusters=neurons.shape[0], init=neurons, n_init=1, max_iter=kmeans_epoch,
                        tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(datas) # k-means training
    mcc = kmeans.cluster_centers_


    #step2:TDP
    mcc, output = TDP(mcc, som.edges)


    #step3:TGM
    pred_all, cluster_num, global_sep, global_com, sep_com = TGM(datas, mcc)


    #result analysis
    end = time.time()
    print('Runtime：%f' % (end - start))
    with open('./datasets/' + dataname + '_label.txt', 'r') as file:
        lines = file.readlines()
    label = [line.strip().split() for line in lines]
    label = np.array(label, dtype=int)
    print(show_result(datas, label, pred_all, cluster_num, global_sep, global_com, sep_com))


