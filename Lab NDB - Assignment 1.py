import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from statistics import mean

def irreducibility(G):
    adj_matrix = nx.to_numpy_array(G)

    final_A = np.identity(len(adj_matrix)) + adj_matrix # I + A

    A_tot = adj_matrix
    for _ in range(2, len(adj_matrix)): # A^2 + ... + A^(n-1)
        A_tot = A_tot.dot(adj_matrix)
        A_tot[A_tot != 0] = 1 # avoid overflow
        final_A = final_A + A_tot

    return (final_A > 0).all() # I + ... + A^(n-1) > 0 

def laplacian(G):
    adj_matrix = nx.to_numpy_array(G)
    
    D = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))

    # fill the correct value for the Laplacian matrix
    for i in range(adj_matrix.shape[0]):
        D[i][i] = sum(adj_matrix[i][:])
    
    L = D - adj_matrix

    eigenvalue = np.linalg.eigvals(L)
    
    return sorted(eigenvalue)[1] > 0 

def bfs(G, v_start):
    distance = { node: float('inf') for node in G.nodes() }

    # create queue
    queue = []

    # init some parameters
    distance[v_start] = 0
    queue.append(v_start)

    # check the connectivity
    while queue:
        current = queue.pop()
        for adj_node in G.neighbors(current):
            if distance[adj_node] == float('inf'):
                distance[adj_node] = distance[current] + 1
                queue.append(adj_node)
    
    connectivity = all(x != float('inf') for x in distance.values())
    return connectivity

def performance(G, method):
    start = time.perf_counter()
    if method == "irreducibility":
        irreducibility(G)
    elif method == "laplacian":
        laplacian(G)
    else:
        bfs(G, list(G.nodes)[0])
    end = time.perf_counter()
    return(end - start)

def complexityMethods(sizes, p, d):
    time_perf_irr = []; time_perf_lapl = []; time_perf_bfs = []

    for nd_size in tqdm(sizes):
        p_ER = nx.erdos_renyi_graph(n = nd_size, p = p) 
        r_regular_graph = nx.random_regular_graph(d = d, n = nd_size, seed=None)

        time_perf_irr.append([performance(p_ER, method = "irreducibility"), performance(r_regular_graph, method = "irreducibility")])
        time_perf_lapl.append([performance(p_ER, method = "laplacian"), performance(r_regular_graph, method = "laplacian")])
        time_perf_bfs.append([performance(p_ER, method = "bfs"), performance(r_regular_graph, method = "bfs")])
    
    return time_perf_irr, time_perf_lapl, time_perf_bfs

def plotPerformance(sizes, time_perf_irr, time_perf_lapl, time_perf_bfs, typegraph, axs):
    method = ["irreducibility", "laplacian", "bfs"]
    
    axs.plot(sizes, time_perf_irr, color = "green", marker = "o")
    axs.plot(sizes, time_perf_lapl, color = "red", marker = "o")
    axs.plot(sizes, time_perf_bfs, color = "blue", marker = "o")
    axs.legend(method)
    axs.set_title(f"Overall performance with {typegraph} Graph")
    axs.set_ylabel("Running Time (sec)")
    axs.set_xlabel("Node sizes")
    axs.grid(color='grey', linestyle='-.')

if __name__ == "__main__":
    n = 10 # choose the number of n nodes
    p = 0.5 # choose the probability of inserting a link between two nodes (used for the Erdos Renyi graph)
    d = 5 # choose the degree of each node (used for the r-regular graph)

    # create erdos-renyi graph
    p_ER = nx.erdos_renyi_graph(n = n, p = p) 
    ax = plt.gca()
    ax.set_title(f'Erdos Renyi graph with {n} nodes and p = {p}')
    nx.draw(p_ER, 				
            font_color="#FFFFFF",
            font_family = 'sans-serif',
            with_labels=True,
            ax = ax)
    plt.savefig('./images/erdosrenyigraph.jpg')
    plt.show()

    # create the r-regular random graph
    r_regular_graph = nx.random_regular_graph(d = d, n = n, seed=None)
    ax = plt.gca()
    ax.set_title(f'r-regular graph with {n} nodes and {d} degree')
    nx.draw(r_regular_graph, 				
        font_color="#FFFFFF",
        font_family = 'sans-serif',
        with_labels=True,
        ax = ax)
    plt.savefig('./images/regulargraph.jpg')
    plt.show()

    # check the connectivity of a given graph
    connectivity_pER = []; connectivity_rG = []

    # check irreducibility method of a graph
    connectivity_pER.append(irreducibility(p_ER))
    connectivity_rG.append(irreducibility(r_regular_graph))
    
    # check laplacian method of a graph
    connectivity_pER.append(laplacian(p_ER))
    connectivity_rG.append(laplacian(r_regular_graph))

    # check BFS method of a graph
    connectivity_pER.append(bfs(p_ER, list(p_ER.nodes)[0]))
    connectivity_rG.append(bfs(r_regular_graph, list(r_regular_graph.nodes)[0]))

    # show the corresponding results of the connectivity for each method
    print(f"The method of irreducibility gives the connectivity {str(connectivity_pER[0]).upper()} for erdos-renyi graph and {str(connectivity_rG[0]).upper()} for r-regular graph")
    print(f"The method of laplacian gives the connectivity {str(connectivity_pER[1]).upper()} for erdos-renyi graph and {str(connectivity_rG[1]).upper()} for r-regular graph")
    print(f"The method of bfs gives the connectivity {str(connectivity_pER[2]).upper()} for erdos-renyi graph and {str(connectivity_rG[2]).upper()} for r-regular graph")
    
    # plotting some plots to compare the complexity of the methods described above
    sizes = [ nd_size for nd_size in range(100, 600, 100)] # the right interval is not considered i.e. 100 to 500 not 600

    time_perf_irr, time_perf_lapl, time_perf_bfs = complexityMethods(sizes, p, d) # get the complexities

    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    plotPerformance(sizes, list(zip(*time_perf_irr))[0], list(zip(*time_perf_lapl))[0], list(zip(*time_perf_bfs))[0], "Erdos Renyi", axs[0])
    plotPerformance(sizes, list(zip(*time_perf_irr))[1], list(zip(*time_perf_lapl))[1], list(zip(*time_perf_bfs))[1], "r-regular", axs[1])
    plt.savefig("./images/performaces.jpg")
    plt.show()

    # Monte Carlo Simulations -  estimates p_c(G) vs p
    nd_size = 100; simulations = 1000
    probs = np.arange(0.1, 1, 0.1)
    est_probs = []
    for p in tqdm(list(probs)):
        true_bfs = [] # count how much the graph is connected
        for _ in range(simulations):
            p_ER = nx.erdos_renyi_graph(n = nd_size, p = p)

            if bfs(p_ER, list(p_ER.nodes)[0]) == True:
                true_bfs.append(1)
            else: 
                true_bfs.append(0)

        est_probs.append(mean(true_bfs))

    plt.plot(list(probs), est_probs, color = "red", marker = "o")
    plt.xlabel("Probabilities")
    plt.ylabel("Estimated connectivity")
    plt.title(f"MC Simulations with {simulations} repetitions - p_c(G) vs p")
    plt.grid(color='grey', linestyle='-.')
    plt.savefig('./images/functionofp.jpg')
    plt.show()

    # Monte Carlo Simulations - estimates p_c(G) vs number of nodes
    est_probs = []; sizes = range(10, 101, 10)
    for nd_size in tqdm(sizes): # bc node <= 100
        true_bfs = [] # count how much the graph is connected
        for _ in range(simulations):
            r_regular_graph_two = nx.random_regular_graph(d = 2, n = nd_size, seed=None)
            r_regular_graph_eight = nx.random_regular_graph(d = 8, n = nd_size, seed=None)

            if bfs(r_regular_graph_two, list(r_regular_graph_two.nodes)[0]) == True:
                if bfs(r_regular_graph_eight, list(r_regular_graph_eight.nodes)[0]) == True:
                    true_bfs.append([1, 1])
                else:
                    true_bfs.append([1, 0])
            elif bfs(r_regular_graph_eight, list(r_regular_graph_two.nodes)[0]) == True: 
                true_bfs.append([0, 1])
            else:
                true_bfs.append([0, 0])

        est_probs.append([ mean(list(zip(*true_bfs))[0]), mean(list(zip(*true_bfs))[1]) ])

    plt.plot(sizes, list(zip(*est_probs))[0], color = "red", marker = "o")
    plt.plot(sizes, list(zip(*est_probs))[1], color = "blue", marker = "o")
    plt.xlabel("Nodes")
    plt.ylabel("Estimated connectivity")
    plt.title(f"MC Simulations with {simulations} repetitions - p_c(G) vs nodes")
    plt.legend(["r = 2", "r = 8"])
    plt.grid(color='grey', linestyle='-.')
    plt.savefig('./images/functionofnodes')
    plt.show()