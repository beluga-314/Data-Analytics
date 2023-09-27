import numpy as np
from scipy.linalg import eigh
import networkx as nx
from matplotlib import pyplot as plt
import os
import time
import pandas as pd
import seaborn as sb

def import_facebook_data(dir):
    edge_list = []
    unique_edges_list = []
    with open(dir, 'r') as file:
        for row in file:
            values = [int(x) for x in row.strip().split()]
            edge_list.append(values)
    unique_edges = set()
    for row in edge_list:
        row = tuple(sorted(row))
        if row not in unique_edges:
            unique_edges.add(row)  
            unique_edges_list.append(row) 
    edge_array = np.array(edge_list)
    return edge_array

def import_bitcoin_data(dir):
    df = pd.read_csv(dir, header=None)
    edge_list = df.iloc[:, :2].values.tolist()
    unique_edges = set()
    unique_edges_list = []
    for row in edge_list:
        row = tuple(sorted(row))
        if row not in unique_edges:
            unique_edges.add(row)  
            unique_edges_list.append(row) 
    nodes = np.unique(unique_edges_list).reshape(-1)
    nodeDict = {element: index for index, element in enumerate(nodes)}
    edge_array = np.array(unique_edges_list)
    for i in range(len(unique_edges_list)):
        for j in range(2):
            edge_array[i][j] = nodeDict[edge_array[i][j]]
    return edge_array


def spectralDecomp_OneIter(edges):
    nodes = np.unique(edges).reshape(-1)
    nodeDict = {element: index for index, element in enumerate(nodes)}
    size = len(nodes)
    Adjacency = np.zeros((size, size)).astype(int)
    for row in edges:
        Adjacency[nodeDict[row[0]], nodeDict[row[1]]] = 1
        Adjacency[nodeDict[row[1]], nodeDict[row[0]]] = 1
    Degree = np.diag(np.sum(Adjacency, axis=1))
    Laplacian = Degree - Adjacency
    eigvals, eigvecs = eigh(Laplacian,Degree)
    sorted_indices = np.argsort(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    fiedler_vector = eigvecs[:, 1]
    unique = np.unique(fiedler_vector >= 0)
    # print(unique)
    if len(unique) != 2:
        return None, None, None
    Community1Head = nodes[np.min(np.where(fiedler_vector < 0)[0])]
    Community2Head = nodes[np.min(np.where(fiedler_vector >= 0)[0])]
    graph_partition = np.zeros((size, 2)).astype(int)
    Column2 = np.where(fiedler_vector>=0, Community2Head, Community1Head)
    graph_partition = np.column_stack((nodes, Column2))
    graph = np.array(graph_partition).astype(int)
    
    return fiedler_vector, Adjacency, graph
    
def spectralDecomposition(edges):
    fiedler_vector, _, graph_partition = spectralDecomp_OneIter(edges)
    if fiedler_vector is None:
        return None
    sorted_fiedler_vector = np.sort(fiedler_vector.reshape(-1))
    consecDiff = np.diff(sorted_fiedler_vector)
    Max = np.max(consecDiff)
    mean = np.mean(consecDiff)
    beta = 200
    if Max < beta * mean:
        return graph_partition
    clusters = np.unique(graph_partition[:,1])
    cluster1Nodes = []
    cluster2Nodes = []
    for row in graph_partition:
        if row[1] == clusters[0]:
            cluster1Nodes.append(row[0])
        else:
            cluster2Nodes.append(row[0])
    cluster1Edges = [[u, v] for u, v in edges if u in cluster1Nodes and v in cluster1Nodes]
    cluster2Edges = [[u, v] for u, v in edges if u in cluster2Nodes and v in cluster2Nodes]
    if len(cluster1Edges) != 0 and len(cluster2Edges) != 0:
        partition1 = spectralDecomposition(cluster1Edges)
        partition2 = spectralDecomposition(cluster2Edges)
        if partition1 is None or partition2 is None:
            return graph_partition
        combined = np.row_stack((partition1, partition2))
        TotalNodes = np.unique(edges).reshape(-1)
        P1Nodes = np.unique(cluster1Edges).reshape(-1)
        P2Nodes = np.unique(cluster2Edges).reshape(-1)
        for node in TotalNodes:
            if node not in P1Nodes and node not in P2Nodes:
                PrevCom = np.where(graph_partition[:,0] == node)[0]
                combined = np.row_stack((combined, graph_partition[PrevCom]))
        return combined

def createSortedAdjMat(graph, edges):
    nodes = np.unique(edges).reshape(-1)
    size = len(nodes)
    Adjacency = np.zeros((size, size)).astype(int)
    for row in edges:
        Adjacency[row[0], row[1]] = 1
        Adjacency[row[1], row[0]] = 1
    indices = graph[:,0][np.argsort(graph[:,1])]
    Adjacency = np.take(Adjacency, indices, axis = 0)
    Adjacency = np.take(Adjacency, indices, axis = 1)
    return Adjacency

def plot(graph, adj, fiedler = None, name = None):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    plot_dir = '../plots/'
    os.makedirs(plot_dir, exist_ok=True)
    if name in ['btc_q1_' ,'fb_q1_']:
        indices = np.argsort(fiedler, axis = 0).reshape(-1)
        fiedler = np.sort(fiedler, axis = 0).reshape(-1)
        plt.figure()
        plt.scatter(x = np.array(range(len(adj))), y = fiedler)
        plt.savefig(plot_dir + name + 'fiedler.png')
        adj = np.take(adj, indices, axis=0)
        adj = np.take(adj, indices, axis=1)
    plt.figure()
    plt.spy(adj)
    plt.savefig(plot_dir + name + 'adjacency.png')
    plt.figure()
    G = nx.Graph(adj)
    pos = nx.spring_layout(G)
    colors = graph[:,1]
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.Set1, with_labels=False)
    plt.savefig(plot_dir + name + 'graph.png')
    del os.environ['QT_QPA_PLATFORM']
    return

def QMerge(node, C, CurrentComm, Adjacency, Degree):
    CommunityNodes = np.where(CurrentComm == C)[0]
    sigma_tot = np.sum(Degree[CommunityNodes])
    k_i_in = 2 * np.sum(Adjacency[node, CommunityNodes])
    k_i = Degree[node]
    return k_i_in - 2*sigma_tot*k_i

def QDeMerge(node, CurrentComm, Adjacency, Degree):
    CommunityNodes = np.where(CurrentComm == CurrentComm[node])[0]
    sigma_tot = np.sum(Degree[CommunityNodes])
    k_i_in = 2 * np.sum(Adjacency[node, CommunityNodes])
    k_i = Degree[node]
    return 2*k_i*sigma_tot - 2*k_i**2 - k_i_in

def Modularity(edges,graph_partition):
        G = nx.Graph()
        G.add_edges_from(edges)
        nodes = np.unique(graph_partition[:,1])
        communities = []
        for i in nodes:
            UniqueComs = set(graph_partition[:,0][np.where(graph_partition[:,1] == i)[0]])
            communities.append(UniqueComs)
        Q = nx.community.modularity(G, communities)
        return Q

def BestComm(node, NeighComms, CurrentComm, QDemerge, Adjacency, Degree):
    Q_Max = 0
    best_community = CurrentComm[node]
    for comm in NeighComms:
        if comm != CurrentComm[node]:
            Qmerge = QMerge(node, comm, CurrentComm, Adjacency, Degree)
            delQ = Qmerge + QDemerge
            if delQ > Q_Max:
                Q_Max = delQ
                best_community = comm
    return best_community, Q_Max

def louvain_one_iter(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    Nodes = G.number_of_nodes()
    Edges = G.number_of_edges()
    nodes = np.arange(Nodes)

    Adjacency = np.zeros((Nodes, Nodes))
    for row in edges:
        Adjacency[row[0], row[1]] = 1
        Adjacency[row[1], row[0]] = 1
    Adjacency = Adjacency / (2 * Edges)
    Degree = np.sum(Adjacency, axis=1)

    Neighbors = []
    for i in nodes:
        Neighbors.append(list(G.neighbors(i)))

    CurrentCommunity = np.arange(Nodes)

    while True:
        changes = 0
        for node in nodes:
            CommAround = np.unique(CurrentCommunity[Neighbors[node]])
            QDM = QDeMerge(node, CurrentCommunity, Adjacency, Degree)
            best_community, Q_Max = BestComm(node, CommAround, CurrentCommunity, QDM, Adjacency, Degree)
            if Q_Max > 0:
                CurrentCommunity[node] = best_community
                changes += 1
        if changes == 0:
            break
    return np.column_stack((nodes, CurrentCommunity))

if __name__ == "__main__":
    
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    print("######## FACEBOOK DATA ########\n")

    fiedler_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    plot(graph_partition_fb,adj_mat_fb,fiedler_vec_fb,'fb_q1_')
    print('Spectral Decomposition\n')
    start_time = time.time()
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    end_time = time.time()
    print('No. of Clusters =', len(np.unique(graph_partition_fb[:,1])))
    print('Clusters are:')
    print(np.unique(graph_partition_fb[:,1]))
    print('Final Modularity of Communities:', Modularity(nodes_connectivity_list_fb, graph_partition_fb))
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds\n")
    
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)
    plot(graph_partition_fb, clustered_adj_mat_fb, name = 'fb_q3_')

    print('Louvain Algorithm\n')
    start_time = time.time()
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    end_time = time.time()
    plot(graph_partition_louvain_fb, createSortedAdjMat(graph_partition_louvain_fb, nodes_connectivity_list_fb), name = 'fb_q4_')
    print('No. of Clusters =', len(np.unique(graph_partition_louvain_fb[:,1])))
    print('Clusters are:')
    print(np.unique(graph_partition_louvain_fb[:,1]))
    print('Final Modularity of Communities:', Modularity(nodes_connectivity_list_fb, graph_partition_louvain_fb))
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")
    print("######## BITCOIN DATA ########\n")
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
    plot(graph_partition_btc,adj_mat_btc,fielder_vec_btc, 'btc_q1_')

    print('Spectral Decomposition\n')
    start_time = time.time()
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    end_time = time.time()
    print('No. of Clusters =', len(np.unique(graph_partition_btc[:,1])))
    print('Clusters are:')
    print(np.unique(graph_partition_btc[:,1]))
    print('Final Modularity of Communities:', Modularity(nodes_connectivity_list_btc, graph_partition_btc))
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)
    plot(graph_partition_btc, clustered_adj_mat_btc, name = 'btc_q3_')
    print('Louvain Algorithm\n')
    start_time = time.time()
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    end_time = time.time()
    plot(graph_partition_louvain_btc, createSortedAdjMat(graph_partition_louvain_btc, nodes_connectivity_list_btc), name = 'btc_q4_')
    print('No. of Clusters =', len(np.unique(graph_partition_louvain_btc[:,1])))
    print('Clusters are:')
    print(np.unique(graph_partition_louvain_btc[:,1]))
    print('Final Modularity of Communities:', Modularity(nodes_connectivity_list_btc, graph_partition_louvain_btc))
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds\n")

def main():
    return
main()