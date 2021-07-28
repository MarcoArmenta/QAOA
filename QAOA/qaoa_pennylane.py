import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

n_wires = 4
dev = qml.device("default.qubit", wires=n_wires, shots=1)
np.random.seed(42)


def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


def U_C(gamma, graph):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)


@qml.qnode(dev)
def circuit(gammas, betas, graph, edge=None, n_layers=1):
    if graph is None:
        print("PASS A GRAPH")
    pauli_z = [[1, 0], [0, -1]]
    pauli_z_2 = np.kron(pauli_z, pauli_z, requires_grad=False)
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i], graph)
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample(comp_basis_measurement(range(n_wires)))
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


def qaoa_maxcut(graph, n_layers=1, lr=0.01, samples=10, optimizer='momentum', epochs=20,
                verbose=False):
    #print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers)
    print('Init:', init_params)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers, graph=graph))
        return neg_obj

    if optimizer.lower() == 'momentum':
        opt = qml.MomentumOptimizer(stepsize=lr)
    elif optimizer.lower() == 'adagrad':
        opt = qml.AdagradOptimizer(stepsize=lr)
    elif optimizer.lower() == 'adam':
        opt = qml.AdamOptimizer(stepsize=lr)
    else:
        opt = qml.GradientDescentOptimizer(stepsize=lr)

    # optimize parameters in objective
    params = init_params
    for i in range(epochs):
        params = opt.step(objective, params)
        if verbose and (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings samples times
    bit_strings = []
    for i in range(0, samples):
        #print(circuit(params[0], params[1], n_layers=n_layers, graph=graph))
        bit_strings.append(int(circuit(params[0], params[1], n_layers=n_layers, graph=graph)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    #print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

    return -objective(params), bit_strings, most_freq_bit_string


def print_from_bitstring(bitstring):
    print(('Solution: {:0' + str(n_wires) + 'b}').format(bitstring))
    return ('{:0' + str(n_wires) + 'b}').format(bitstring)


if __name__ == '__main__':

    #graph1 = [(0, 1), (1, 2), (2, 3), (3, 4),
    #          (1, 4), (0, 3), (2, 4)]

    graph2 = [(0, 1), (1, 2), (2, 3), (3, 0),
              (0, 2), (3, 1)]

    graph = graph2

    G = nx.Graph()
    G.add_nodes_from(list(range(n_wires)))
    G.add_edges_from(graph)
    #subax1 = plt.subplot(121)

    colors = ['r' for _ in G.nodes()]
    default_axes = plt.axes(frameon=False)
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(graph)
    #pos = nx.planar_layout(G)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
    #plt.show()

    num_layers = 2

    #qaoa = QAOA(n_wires, graph, num_layers)
    #_, _, bitstring = qaoa.maxcut()[2]

    bitstring = qaoa_maxcut(graph=graph, n_layers=num_layers, epochs=60)[2]
    #print(bitstring2)
    #print(str(bitstring2))
    x = print_from_bitstring(bitstring)
    colors = ['r' if x[i] == '1' else 'b' for i in range(len(x))]

    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
    plt.show()

    # perform qaoa on our graph with p=1,2 and
    # keep the bitstring sample lists
    #bitstrings1 = qaoa_maxcut(graph=graph, n_layers=3, verbose=True, optimizer='adam')[2]
    '''
    if n_wires == 4:
        print('Solution: {:04b}'.format(bitstrings1))

    if n_wires == 5:
        print('Solution: {:05b}'.format(bitstrings1))

    if n_wires == 6:
        print('Solution: {:06b}'.format(bitstrings1))
    '''
    #bitstrings2 = qaoa_maxcut(n_layers=2)[2]
    #print('Solution: {:05b}'.format(bitstrings2))


    #xticks = range(0, 16)
    #xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
    #bins = np.arange(0, 17) - 0.5

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    #plt.subplot(1, 2, 1)
    #plt.title("n_layers=1")
    #plt.xlabel("bitstrings")
    #plt.ylabel("freq.")
    #plt.xticks(xticks, xtick_labels, rotation="vertical")
    #plt.hist(bitstrings1, bins=bins)
    #plt.subplot(1, 2, 2)
    #plt.title("n_layers=2")
    #plt.xlabel("bitstrings")
    #plt.ylabel("freq.")
    #plt.xticks(xticks, xtick_labels, rotation="vertical")
    #plt.hist(bitstrings2, bins=bins)
    #plt.tight_layout()
    #plt.show()
