from ase.io import read as aseread
from matscipy.neighbours import neighbour_list
import networkx as nx


def atoms_to_nxgraph(atoms, cutoff):
    ni, nj = neighbour_list('ij', atoms, cutoff)
    adjacency_matrix = np.zeros((len(atoms), len(atoms))).astype(np.int)
    for i, j in zip (ni, nj):
        adjacency_matrix[i,j] = 1
    graph = nx.from_numpy_matrix(np.array(adjacency_matrix))
    return graph

def minimal_cycles(graph, cutoff=9):
    all_cycles = []
    for node in graph.nodes():
        # Avoid A-B-A cycles with len(p) > 3. Avoid large non-minimal cycles with cutoff=9
        cycles = [set(p) for p in nx.algorithms.all_simple_paths(graph, node, node, cutoff=cutoff) if len(p) > 3]
        for cycle in cycles:
            if cycle not in all_cycles:
                all_cycles.append(cycle)
    # purge non minimal cycles and non-cycles
    for c0 in [cycle for cycle in all_cycles if len(cycle) > 6]:
        sub = nx.Graph(graph.subgraph(list(c0)))
        if sub.number_of_edges() != sub.number_of_nodes():
            all_cycles.remove(c0)
    return all_cycles

def cycle_dual_graph(all_cycles):
    # create the network of connected cycles
    cycle_adjacency = np.zeros((len(all_cycles), len(all_cycles))).astype(np.int)
    for i, ci in enumerate(all_cycles):
        for j, cj in enumerate(all_cycles):
            if j > i:
                if len(ci.intersection(cj)) > 0:
                    cycle_adjacency[i,j] = 1
    cycle_adjacency += cycle_adjacency.T
    # create the dual network: a node is a minimal ring, an edge is a shared edge between two rings (e.g. an O atom in the real system)
    graph_dual = nx.from_numpy_matrix(cycle_adjacency)
    return graph_dual

########################################################################
########################################################################

at = aseread('../SiO2_bilayer.xyz', format='extxyz')

at.positions[:,2] -= at.get_positions()[:,2].mean()
indices = np.where(np.logical_and(at.positions[:,2] > .2, at.get_atomic_numbers() == 14))[0]
atsiup = at[indices]
atsiup.write('atsiup.xyz', format='extxyz')

cutoff = 3.8 # 2 * 1.6 + some extra for elongation. visual inspection first!
graph = atoms_to_nxgraph(atsiup, cutoff)

all_cycles = minimal_cycles(graph, cutoff=9)

graph_dual = cycle_dual_graph(all_cycles)

