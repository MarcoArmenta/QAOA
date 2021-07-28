import matplotlib.pyplot as plt
import networkx as nx

from QAOA.QAOA import QAOA_01

if __name__ == '__main__':
    graph2 = [(0, 1), (1, 2), (2, 3), (3, 0),
              (0, 2), (3, 1)]
    n2 = 4

    graph3 = [(0, 1), (1, 2)]
    n3 = 3

    graph4 = [(0, 1), (1, 2), (2, 3), (3, 4),
              (1, 4), (0, 3), (2, 4)]
    n4 = 5

    graph = graph4
    n = n4

    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    G.add_edges_from(graph)

    qaoa = QAOA_01(G)


