import functools
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from ortools.sat.python import cp_model
from qiskit_aer.primitives import Estimator as LocalEstimator
from qiskit_aer.primitives import Sampler as LocalSampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============== Graph Generators ==============

def generate_generalised_petersen(n: int, k: int = 2, weight: float = 1.0, draw: bool = False) -> rx.PyGraph:
    """Generalised Petersen graph with random outer edge weights."""
    if n % 2 != 0:
        raise ValueError("n must be even (n = 2 * outer_cycle_size).")
    m = n // 2
    if not (1 <= k < m):
        raise ValueError(f"k must satisfy 1 <= k < {m} (got k={k}).")

    G = rx.PyGraph()
    G.add_nodes_from(range(n))

    rng = np.random.default_rng(0)
    outer = [(i, (i+1)%m, 0.5 + rng.random()*1.0) for i in range(m)]
    inner = [(m + i, m + ((i + k) % m), weight) for i in range(m)]
    spokes = [(i, m + i, weight) for i in range(m)]

    G.add_edges_from(outer + inner + spokes)

    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()

    return G


def generate_random_regular(n: int, degree: int = 3, draw: bool = False) -> rx.PyGraph:
    """Random regular graph with uniform weights."""
    if n * degree % 2 != 0:
        raise ValueError("n * degree must be even.")
    
    G = rx.generators.random_regular_graph(n, degree, seed=42)
    
    # Add uniform weights
    for edge_idx in range(len(G.edge_list())):
        G.update_edge_by_index(edge_idx, 1.0)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_erdos_renyi(n: int, p: float = 0.5, draw: bool = False) -> rx.PyGraph:
    """Erdos-Renyi random graph with random weights."""
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                weight = 0.5 + rng.random() * 1.0
                edges.append((i, j, weight))
    
    G.add_edges_from(edges)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_cycle(n: int, draw: bool = False) -> rx.PyGraph:
    """Simple cycle graph with uniform weights."""
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    edges = [(i, (i+1) % n, 1.0) for i in range(n)]
    G.add_edges_from(edges)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_sparse_tree_like(n: int, degree: int = 3, draw: bool = False) -> rx.PyGraph:
    """
    Sparse locally-tree-like random graph.
    Uses configuration model with low degree to create sparse graphs
    that locally resemble trees (few short cycles).
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    
    # Create degree sequence (approximately regular but allows variation)
    degree_seq = [degree] * n
    if sum(degree_seq) % 2 != 0:
        degree_seq[0] += 1
    
    # Configuration model: create stubs and pair them randomly
    stubs = []
    for node, d in enumerate(degree_seq):
        stubs.extend([node] * d)
    
    rng.shuffle(stubs)
    
    edges_added = set()
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v and (min(u, v), max(u, v)) not in edges_added:
            weight = 0.5 + rng.random() * 1.0
            G.add_edge(u, v, weight)
            edges_added.add((min(u, v), max(u, v)))
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_stochastic_block_model(n: int, num_communities: int = 2, p_in: float = 0.7, p_out: float = 0.1, draw: bool = False) -> rx.PyGraph:
    """
    Stochastic Block Model (SBM) graph.
    Creates communities with dense intra-community edges and sparse inter-community edges.
    
    Args:
        n: Total number of nodes
        num_communities: Number of communities
        p_in: Probability of edge within same community
        p_out: Probability of edge between different communities
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    
    # Assign nodes to communities
    community_size = n // num_communities
    communities = [i // community_size for i in range(n)]
    # Handle remainder
    for i in range(n % num_communities):
        communities[-(i+1)] = num_communities - 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if communities[i] == communities[j] else p_out
            if rng.random() < p:
                weight = 0.5 + rng.random() * 1.0
                edges.append((i, j, weight))
    
    G.add_edges_from(edges)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_small_world(n: int, k: int = 4, p_rewire: float = 0.1, draw: bool = False) -> rx.PyGraph:
    """
    Watts-Strogatz small-world graph.
    Starts with a ring lattice and randomly rewires edges.
    
    Args:
        n: Number of nodes
        k: Each node connects to k nearest neighbors (must be even)
        p_rewire: Probability of rewiring each edge
    """
    if k % 2 != 0:
        k += 1  # Ensure k is even
    
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    
    # Create ring lattice
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            edges.append((i, target))
    
    # Rewire edges
    rewired_edges = []
    for u, v in edges:
        if rng.random() < p_rewire:
            # Rewire to random node
            new_v = rng.integers(0, n)
            attempts = 0
            while new_v == u or (min(u, new_v), max(u, new_v)) in [(min(e[0], e[1]), max(e[0], e[1])) for e in rewired_edges]:
                new_v = rng.integers(0, n)
                attempts += 1
                if attempts > 100:
                    new_v = v  # Keep original if can't find new target
                    break
            v = new_v
        weight = 0.5 + rng.random() * 1.0
        rewired_edges.append((u, v, weight))
    
    # Remove duplicates
    seen = set()
    for u, v, w in rewired_edges:
        edge_key = (min(u, v), max(u, v))
        if edge_key not in seen and u != v:
            G.add_edge(u, v, w)
            seen.add(edge_key)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_power_law(n: int, m: int = 2, draw: bool = False) -> rx.PyGraph:
    """
    Barabási-Albert preferential attachment (scale-free/power-law) graph.
    
    Args:
        n: Number of nodes
        m: Number of edges to attach from new node to existing nodes
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    
    # Start with a small complete graph
    initial_nodes = m + 1
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            weight = 0.5 + rng.random() * 1.0
            G.add_edge(i, j, weight)
    
    # Track degrees for preferential attachment
    degrees = [m] * initial_nodes + [0] * (n - initial_nodes)
    
    # Add remaining nodes with preferential attachment
    for new_node in range(initial_nodes, n):
        # Calculate probabilities based on degree
        total_degree = sum(degrees[:new_node])
        if total_degree == 0:
            probs = [1.0 / new_node] * new_node
        else:
            probs = [d / total_degree for d in degrees[:new_node]]
        
        # Choose m nodes to connect to
        targets = set()
        while len(targets) < min(m, new_node):
            r = rng.random()
            cumsum = 0
            for node, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    targets.add(node)
                    break
        
        for target in targets:
            weight = 0.5 + rng.random() * 1.0
            G.add_edge(new_node, target, weight)
            degrees[new_node] += 1
            degrees[target] += 1
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


def generate_weighted_similar_neighborhoods(n: int, k: int = 4, draw: bool = False) -> rx.PyGraph:
    """
    Graph with similar local neighborhoods and structured weights.
    Creates a graph where nodes have similar local structure,
    making it challenging for QAOA due to symmetry.
    
    Args:
        n: Number of nodes
        k: Neighborhood size (connections per node)
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(n))
    
    rng = np.random.default_rng(42)
    
    # Create a circulant-like graph with consistent neighborhoods
    edges = set()
    for i in range(n):
        for offset in range(1, k // 2 + 1):
            j = (i + offset) % n
            edges.add((min(i, j), max(i, j)))
    
    # Add some random edges to break perfect symmetry slightly
    num_random = n // 4
    for _ in range(num_random):
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    
    # Assign weights based on node indices to create structure
    for u, v in edges:
        # Weight depends on "distance" in the ring
        dist = min(abs(u - v), n - abs(u - v))
        base_weight = 1.0 / (1 + dist * 0.1)
        weight = base_weight + rng.random() * 0.3
        G.add_edge(u, v, weight)
    
    if draw:
        draw_graph(G, node_size=600, with_labels=True)
        plt.show()
    
    return G


GRAPH_GENERATORS = {
    "petersen": generate_generalised_petersen,
    "regular": generate_random_regular,
    "erdos-renyi": generate_erdos_renyi,
    "cycle": generate_cycle,
    "tree-like": generate_sparse_tree_like,
    "sbm": generate_stochastic_block_model,
    "small-world": generate_small_world,
    "power-law": generate_power_law,
    "similar-neighborhoods": generate_weighted_similar_neighborhoods,
}

# ============== QAOA Core Functions ==============


def run_on_IBM_backend(n: int):
    service = QiskitRuntimeService()
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=n,
    )
    return backend


def cmax_ortools_exact(rx_graph):
    n = len(list(rx_graph.nodes()))
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    y = {}

    for u, v in rx_graph.edge_list():
        y[(u,v)] = model.NewBoolVar(f"y_{u}_{v}")
        model.Add(y[(u,v)] >= x[u] - x[v])
        model.Add(y[(u,v)] >= x[v] - x[u])
        model.Add(y[(u,v)] <= x[u] + x[v])
        model.Add(y[(u,v)] <= 2 - x[u] - x[v])
    SCALE = 1000
    model.Maximize(sum(int(round(SCALE*float(rx_graph.get_edge_data(u,v)))) * y[(u,v)]
                       for u,v in rx_graph.edge_list()))
    solver = cp_model.CpSolver()
    solver.Solve(model)
    cmax = solver.ObjectiveValue() / SCALE
    return cmax


def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list


def build_cost_and_circuit(graph, n, reps=2):
    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)
    # print("Cost Function Hamiltonian:\n", cost_hamiltonian)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    circuit.measure_all()
    # circuit.draw("mpl")
    return cost_hamiltonian, circuit


def optimize_for_backend(circuit, backend):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit)
    # candidate_circuit.draw("mpl", fold=False, idle_wires=False)
    return candidate_circuit

objective_func_vals = [] 
print = functools.partial(print, flush=True)


def cost_func_estimator_IBM(params, ansatz, hamiltonian, estimator):

    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    result = job.result()[0]
    cost = result.data.evs
    objective_func_vals.append(cost)
    return cost


def cost_func_estimator_local(params, ansatz, hamiltonian, estimator):

    job = estimator.run(ansatz, hamiltonian, params)
    result = job.result()
    cost = result.values[0]
    objective_func_vals.append(cost)
    return cost


def run_optimization_local(candidate_circuit, cost_hamiltonian, backend, init_params):
    estimator = LocalEstimator()
    result = minimize(
        cost_func_estimator_local, 
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    return result


def run_optimization_IBM(candidate_circuit, cost_hamiltonian, backend, init_params):
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 1000
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

        result = minimize(
            cost_func_estimator_IBM,
            init_params,
            args=(candidate_circuit, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
        )
    return result


def sample_distribution_local(optimized_circuit, backend, shots=10_000):
    sampler = LocalSampler()
    job = sampler.run(optimized_circuit, shots=shots) 
    result = job.result()
    counts_int = result.quasi_dists[0]
    counts_int = {k: int(v * shots) for k, v in counts_int.items()}
    total = sum(counts_int.values())
    final_distribution_int = {k: v / total for k, v in counts_int.items()}
    return final_distribution_int


def sample_distribution_IBM(optimized_circuit, backend, shots=10_000):
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        sampler.options.default_shots = shots 
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.twirling.enable_gates = True    
        job = sampler.run([optimized_circuit])
        counts_int = job.result()[0].data.meas.get_int_counts()

    total = sum(counts_int.values())
    final_distribution_int = {k: v / total for k, v in counts_int.items()}
    return final_distribution_int


def to_bitstring(integer, num_bits):
    return [int(digit) for digit in np.binary_repr(integer, width=num_bits)]


def most_likely_bitstring_from_dist(final_distribution_int, num_bits):
    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    most_likely = keys[np.argmax(np.abs(values))]
    bitstring = to_bitstring(most_likely, num_bits)
    bitstring.reverse()
    return bitstring


def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(
        list(graph.nodes())
    ), "The length of x must coincide with the number of nodes in the graph."
    return sum(
        x[u] * (1 - x[v]) + x[v] * (1 - x[u])
        for u, v in list(graph.edge_list())
    )


def run_qaoa(n, graph, depth, initial_gamma, initial_beta, using_IBM):

    # Build cost + circuit
    cost_hamiltonian, circuit = build_cost_and_circuit(graph, n, reps=depth)

    if (using_IBM):
        backend = run_on_IBM_backend(n)
    else:
        backend = AerSimulator()
    
    # Transpile for backend
    candidate_circuit = optimize_for_backend(circuit, backend)

    # Initial params (same as yours)

    init_params = [initial_beta] * depth + [initial_gamma] * depth

    if (using_IBM):
        result = run_optimization_IBM(candidate_circuit, cost_hamiltonian, backend, init_params)
    else:
        result = run_optimization_local(candidate_circuit, cost_hamiltonian, backend, init_params)

    # Sample final circuit
    optimized_circuit = candidate_circuit.assign_parameters(result.x)

    if (using_IBM):
        final_distribution_int = sample_distribution_IBM(optimized_circuit, backend, shots=10_000)
    else:
        final_distribution_int = sample_distribution_local(optimized_circuit, backend, shots=10_000)

    # Most likely bitstring
    most_likely_bitstring = most_likely_bitstring_from_dist(final_distribution_int, len(graph))    
    
    cut_value = evaluate_sample(most_likely_bitstring, graph)
    return cut_value


def max_weighted_degree(graph: rx.PyGraph) -> float:
    return max(
        (
            sum(abs(float(graph.get_edge_data(i, j))) for j in graph.neighbors(i))
            for i in graph.node_indices()
        ),
        default=1.0,
    )




def run_grid_search(graph, n, using_IBM, num_of_zooms):

    zoom_factor = 0.6

    # grid search initial param values
    depths = [2, 3, 4]
    initial_betas = np.array([np.pi/12, np.pi/8, np.pi/6, np.pi/4, 3*np.pi/8, np.pi/2])
    initial_gammas = np.pi / 3 * np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])  # scaled for cubic graph

    # graph = generate_graph(n)          
    c_max = cmax_ortools_exact(graph)

    print(f"C_MAX (classical optimal): {c_max}")
    print(f"Grid size: {6}x{6}x{len(depths)} = {6*6*len(depths)} combinations per zoom\n")

    for zoom_idx in range(num_of_zooms):
        print(f"=================== Zoom Level {zoom_idx} ===================")

        best_accuracy = -1.0
        best_beta, best_gamma, best_depth = None, None, None

        for initial_gamma in initial_gammas:
            for initial_beta in initial_betas:
                print(f"  Testing: γ={initial_gamma:.4f}, β={initial_beta:.4f}")

                for depth in depths:
                    value = run_qaoa(n, graph, depth, initial_gamma, initial_beta, using_IBM)/c_max
                    
                    if value > best_accuracy:
                            best_accuracy = value - 0.05
                            best_beta, best_gamma, best_depth = initial_beta, initial_gamma, depth
            
        
        print(f"\n  ✓ Best so far: β={best_beta:.4f}, γ={best_gamma:.4f}, depth={best_depth}, accuracy={best_accuracy:.4f}\n")

        # linear zoom around the best β, γ
        beta_range = best_beta * zoom_factor
        gamma_range = best_gamma * zoom_factor
        
        beta_min = max(best_beta - beta_range/2, 0)
        beta_max = min(best_beta + beta_range/2, np.pi/2)
        gamma_min = max(best_gamma - gamma_range/2, 0)
        gamma_max = min(best_gamma + gamma_range/2, np.pi)
        
        # refine 6×6 grid around the best region (linear spacing)
        initial_betas = np.linspace(beta_min, beta_max, 6)
        initial_gammas = np.linspace(gamma_min, gamma_max, 6)   
        depths = [best_depth]  # lock depth

    return best_beta, best_gamma, best_depth, best_accuracy, c_max

    

# ============== CLI ==============

def create_parser():
    parser = argparse.ArgumentParser(
        prog="qaoa_maxcut",
        description="""
╔══════════════════════════════════════════════════════════════════╗
║                    QAOA MaxCut Solver                            ║
║  Solves the MaxCut problem using Quantum Approximate             ║
║  Optimization Algorithm with adaptive grid search.               ║
╚══════════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --graph petersen --nodes 20 --backend local --zooms 3
  %(prog)s --graph erdos-renyi --nodes 10 --backend ibm --zooms 2
  %(prog)s --graph cycle --nodes 12 --backend local --zooms 5 --draw

Graph Types:
  petersen              Generalised Petersen graph (nodes must be even)
  regular               Random 3-regular graph
  erdos-renyi           Erdős-Rényi G(n,p) random graph (p=0.5)
  cycle                 Simple cycle graph
  tree-like             Sparse locally-tree-like graph
  sbm                   Stochastic Block Model (2 communities)
  small-world           Watts-Strogatz small-world graph
  power-law             Barabási-Albert scale-free graph
  similar-neighborhoods Weighted graph with similar local structure
        """
    )
    
    parser.add_argument(
        "-g", "--graph",
        type=str,
        choices=list(GRAPH_GENERATORS.keys()),
        required=True,
        help="Type of graph to generate"
    )
    
    parser.add_argument(
        "-n", "--nodes",
        type=int,
        required=True,
        help="Number of nodes in the graph"
    )
    
    parser.add_argument(
        "-b", "--backend",
        type=str,
        choices=["local", "ibm"],
        default="local",
        help="Execution backend: 'local' (simulator) or 'ibm' (real quantum computer)"
    )
    
    parser.add_argument(
        "-z", "--zooms",
        type=int,
        default=3,
        help="Number of zoom iterations (default: 3)"
    )
    
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw the graph before solving"
    )
    
    return parser 
    

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if args.nodes < 4:
        print("Error: Number of nodes must be at least 4")
        sys.exit(1)
    
    if args.graph == "petersen" and args.nodes % 2 != 0:
        print("Error: Petersen graph requires an even number of nodes")
        sys.exit(1)
        
    # Generate graph
    print(f"\nGenerating {args.graph} graph with {args.nodes} nodes...")
    
    graph_generator = GRAPH_GENERATORS[args.graph]
    
    # Handle drawing separately to avoid threading issues
    if args.draw:
        graph = graph_generator(args.nodes, draw=False)
        draw_graph(graph, node_size=600, with_labels=True)
        plt.savefig("graph.png")
        print("Graph saved to graph.png")
        plt.clf()
    else:
        graph = graph_generator(args.nodes, draw=False)
    
    print(f"Edges: {len(graph.edge_list())}")
    print(f"Backend: {'IBM Quantum' if args.backend == 'ibm' else 'Local Simulator'}")
    print(f"Zoom levels: {args.zooms}")

    
    # Run QAOA
    using_IBM = (args.backend == "ibm")
    
    best_beta, best_gamma, best_depth, best_accuracy, c_max = run_grid_search(
        graph=graph,
        n=args.nodes,
        using_IBM=using_IBM,
        num_of_zooms=args.zooms,
    )
    
    # Print results
    print("                    RESULTS")
    print(f"  Classical optimal (C_MAX):  {c_max:.4f}")
    print(f"  QAOA approximation ratio:   {best_accuracy:.4f}")
    print(f"  Best β (beta):              {best_beta:.6f}")
    print(f"  Best γ (gamma):             {best_gamma:.6f}")
    print(f"  Best depth:                 {best_depth}")
   
if __name__ == "__main__":
    main()