import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import Counter
import networkx as nx
import community
from itertools import permutations 
import matplotlib.ticker as mtick
import matplotlib as mpl
import matplotlib.cm as cm
import sys
sys.path.append("../../../LocalGraphClustering/")
import localgraphclustering as lgc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.csgraph import connected_components
import pickle
import logging

"""
G: GraphLocal instance
S: seed region where the local embedding needs to be computed
x,y: global embedding coordinates
ntrials: number of random seeds to select from S
seeds_lb,seeds_ub: the lower bound and the upper bound of the number of seeds to use in each experiment,
                   the code will use binary search to find the first radius threshold such that both lower 
                   bound and upper bound can be met.
scale: scaling parameter for global embedding when concatenating global coords with clustering results in SVD, 
       larger value means less deviated from global embedding.
"""

from scipy.sparse.csgraph import connected_components
def largest_cc(G,S):
    subgraph = G.adjacency_matrix[S,:][:,S]
    ret = connected_components(subgraph,directed=False)
    l = Counter(ret[1]).most_common(1)[0][0]
    S = [S[i] for i in range(len(S)) if ret[1][i] == l]
    return S
    


def local_embedding(G,S,x,y,method="sl_weighted",nprocs=40,ntrials=500,seeds_lb=100,seeds_ub=1000,delta=0.1,
    alpha=0.01,iterations=1000000,rho=1.0e-6,scale=400):
    records = -10*np.ones((G._num_vertices,ntrials))
    #records_flow = -10*np.zeros((G._num_vertices,ntrials))
    if method == "sl_weighted":
        logname = "sl_weighted_002_1_node_2.log"
    elif method == "mqi_weighted":
        logname = "mqi_weighted_002_1_node_2.log"
    elif method == "l1reg-rand":
        logname = "l1reg-rand_002_1_node_2.log"
    logging.basicConfig(filename=logname, filemode='w', level=logging.INFO, format="%(message)s")
    def wrapper(q_in,q_out):
        while True:
            trial_id = q_in.get()
            if trial_id is None:
                break
            np.random.seed(trial_id)
            seeds_subset = np.random.choice(S,1)
            seeds_subset = seed_grow_bfs_steps(G,seeds_subset,3)
            seeds_subset = largest_cc(G,seeds_subset)
            print(len(seeds_subset))
            if method == "l1reg-rand":
                output,vals = lgc.approximate_PageRank(G,seeds_subset,normalize=False,rho=rho,method='l1reg-rand',alpha=alpha,iterations=iterations)
                sc = lgc.sweep_cut(G,(output,vals))
                print(trial_id,len(seeds_subset),len(output),G.compute_conductance(seeds_subset),len(sc[0]),sc[1])
                log = ", ".join([str(trial_id),str(len(seeds_subset)),str(len(output)),str(G.compute_conductance(seeds_subset)),str(len(sc[0])),str(sc[1])])
                pos_ids = np.nonzero(vals>0)[0]
                output = output[pos_ids]
                vals = vals[pos_ids]
                vals = np.log10(vals/np.max(vals))
            elif method == "sl_weighted":
                output = lgc.flow_clustering(G,seeds_subset,method="sl_weighted",delta=delta)[0]
                print(trial_id,len(seeds_subset),len(output),G.compute_conductance(seeds_subset),G.compute_conductance(output))
                log = ", ".join([str(trial_id),str(len(seeds_subset)),str(len(output)),str(G.compute_conductance(seeds_subset)),str(G.compute_conductance(output))])
                vals = np.zeros(len(output))
            elif method == "mqi_weighted":
                output = lgc.flow_clustering(G,seeds_subset,method="mqi_weighted")[0]
                print(trial_id,len(seeds_subset),len(output),G.compute_conductance(seeds_subset),G.compute_conductance(output))
                log = ", ".join([str(trial_id),str(len(seeds_subset)),str(len(output)),str(G.compute_conductance(seeds_subset)),str(G.compute_conductance(output))])
                vals = np.zeros(len(output))
            q_out.put((trial_id,output,vals,log))
    q_in,q_out = mp.Queue(),mp.Queue()
    for i in range(ntrials):
        q_in.put(i)
    for _ in range(nprocs):
        q_in.put(None)
    procs = [mp.Process(target=wrapper,args=(q_in,q_out)) for _ in range(nprocs)]
    for p in procs:
        p.start()
    ncounts = 0
    visited = set()
    while ncounts < ntrials:
        trial_id,output,vals,log = q_out.get()
        visited = visited.union(output)
        print(len(visited))
        logging.info(log+"\n")
        records[output,trial_id] = vals
        ncounts += 1
    for p in procs:
        p.join()
#     svd = TruncatedSVD(n_components=2, random_state=42,n_iter=40)
#     records_local = np.hstack([records[:,0:ntrials],np.array([scale*x,scale*y]).T])
#     svd.fit(records_local)
#     local_coords = svd.transform(records_local)
#     local_x,local_y = local_coords[:,0],local_coords[:,1]
    return records

import queue

def seed_grow_bfs_steps(g,seeds,steps):
    """
    grow the initial seed set through BFS until its size reaches 
    a given ratio of the total number of nodes.
    """
    Q = queue.Queue()
    visited = np.zeros(g._num_vertices)
    visited[seeds] = 1
    for s in seeds:
        Q.put(s)
    if isinstance(seeds,np.ndarray):
        seeds = seeds.tolist()
    else:
        seeds = list(seeds)
    for step in range(steps):
        for k in range(Q.qsize()):
            node = Q.get()
            si,ei = g.adjacency_matrix.indptr[node],g.adjacency_matrix.indptr[node+1]
            neighs = g.adjacency_matrix.indices[si:ei]
            for i in range(len(neighs)):
                if visited[neighs[i]] == 0:
                    visited[neighs[i]] = 1
                    seeds.append(neighs[i])
                    Q.put(neighs[i])
    return seeds



import sys
sys.path.append("../../../LocalGraphClustering/")
import localgraphclustering as lgc

G = lgc.GraphLocal("../../dataset/lawlor-spectra-k32.edgelist","edgelist")

import pandas as pd
from sklearn.neighbors import NearestNeighbors

df = pd.read_table("../../dataset/lawlor-spectra-k32.coords",header=None)
coords = df[[0,1]].values
coords[:,1] *= 4
coords[:,0] *= 10
x,y = -1*coords[:,0],-1*coords[:,1]

S = np.nonzero((x>0.002))[0]
records_flow = local_embedding(G,S,x,y,ntrials=500,delta=1.0,nprocs=120)

wptr = open("records_flow_002_1_node_3.p","wb")
pickle.dump(records_flow,wptr)
wptr.close()

X = sp.csr_matrix(records_flow+10)
X.eliminate_zeros()

wptr = open("flow_3_bfs_delta_1.p","wb")
pickle.dump(X,wptr)
wptr.close()

S = np.nonzero((x>0.002))[0]
records_flow = local_embedding(G,S,x,y,ntrials=500,delta=1.0,nprocs=120)

wptr = open("records_flow_002_1_node_2.p","wb")
pickle.dump(records_flow,wptr)
wptr.close()

X = sp.csr_matrix(records_flow+10)
X.eliminate_zeros()

wptr = open("flow_3_bfs_delta_0.2.p","wb")
pickle.dump(X,wptr)
wptr.close()

S = np.nonzero((x>0.002))[0]
records_flow = local_embedding(G,S,x,y,ntrials=500,delta=1.0,nprocs=120)

wptr = open("records_flow_002_1_node_1.p","wb")
pickle.dump(records_flow,wptr)
wptr.close()

X = sp.csr_matrix(records_flow+10)
X.eliminate_zeros()

wptr = open("flow_3_bfs_delta_0.1.p","wb")
pickle.dump(X,wptr)
wptr.close()

records_spectral = local_embedding(G,S,x,y,ntrials=1000,method="l1reg-rand",nprocs=80,alpha=0.1,rho=1.0e-7,iterations=int(1.0e7))

wptr = open("records_spectral_002_1_node_2.p","wb")
pickle.dump(records_spectral,wptr)
wptr.close()

X = sp.csr_matrix(records_spectral+10)
X.eliminate_zeros()

wptr = open("embeddings/spectral_3_bfs.p","wb")
pickle.dump(X,wptr)
wptr.close()

records_mqi = local_embedding(G,S,x,y,ntrials=500,method="mqi_weighted",nprocs=80)

wptr = open("records_mqi_002_1_node_1.p","wb")
pickle.dump(records_mqi,wptr)
wptr.close()

X = sp.csr_matrix(records_mqi+10)
X.eliminate_zeros()

wptr = open("embeddings/mqi_3_bfs.p","wb")
pickle.dump(X,wptr)
wptr.close()

## For next, please run the scripts inside "cond_hists" to get the figures