import matplotlib
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from ortools.sat.python import cp_model



def generate_graph(n: int, k: int = 2, weight: float = 1.0, draw: bool = True) -> rx.PyGraph:
    if n % 2 != 0:
        raise ValueError("n must be even (n = 2 * outer_cycle_size).")
    m = n // 2
    if not (1 <= k < m):
        raise ValueError(f"k must satisfy 1 <= k < {m} (got k={k}).")

    G = rx.PyGraph()
    G.add_nodes_from(range(n))

    rng = np.random.default_rng(0)
    outer = [(i, (i+1)%m, 0.5 + rng.random()*1.0) for i in range(m)]

    inner = [ (m + i, m + ((i + k) % m), weight) for i in range(m) ]

    spokes = [ (i, m + i, weight) for i in range(m) ]

    G.add_edges_from(outer + inner + spokes)

    if draw:
        draw_graph(G, node_size=600, with_labels=True)

    return G

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
    circuit.draw("mpl")
    return cost_hamiltonian, circuit


def optimize_for_backend(circuit, backend):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit)
    candidate_circuit.draw("mpl", fold=False, idle_wires=False)
    return candidate_circuit


objective_func_vals = []  # keep your global trace

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    result = job.result()[0]
    cost = result.data.evs
    objective_func_vals.append(cost)
    return cost


def run_optimization(candidate_circuit, cost_hamiltonian, backend, init_params):
    estimator = Estimator(mode=backend)
    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    return result


def sample_distribution(optimized_circuit, backend, shots=10_000):
    sampler = Sampler(mode=backend)
    job = sampler.run([(optimized_circuit,)], shots=shots)
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


def plot_convergence(trace):
    plt.figure(figsize=(12, 6))
    plt.plot(trace)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(
        list(graph.nodes())
    ), "The length of x must coincide with the number of nodes in the graph."
    return sum(
        x[u] * (1 - x[v]) + x[v] * (1 - x[u])
        for u, v in list(graph.edge_list())
    )

def run_qaoa(n, graph, depth, initial_gamma, initial_beta):



    # Build cost + circuit
    cost_hamiltonian, circuit = build_cost_and_circuit(graph, n, reps=depth)

    # Backend
    backend = AerSimulator()
    # print(f"Using backend: {backend}")

    # Transpile for backend
    candidate_circuit = optimize_for_backend(circuit, backend)

    # Initial params (same as yours)

    init_params = [initial_beta] * depth + [initial_gamma] * depth

    # Optimize
    result = run_optimization(candidate_circuit, cost_hamiltonian, backend, init_params)
    # print("\nOptimization result:\n", result)

    # Plot convergence
    # plot_convergence(objective_func_vals)

    # Sample final circuit
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    final_distribution_int = sample_distribution(optimized_circuit, backend, shots=10_000)
    # print("\nFinal distribution:\n", final_distribution_int)

    # Most likely bitstring
    most_likely_bitstring = most_likely_bitstring_from_dist(final_distribution_int, len(graph))
    # print("\nMost likely bitstring:", most_likely_bitstring)
    
    
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



def main():

    num_of_zooms = 5
    # Grid search initial param values
    depths = [2, 3, 4]
    initial_betas = np.array([np.pi/12, np.pi/8, np.pi/6, np.pi/4, 3*np.pi/8, np.pi/2])
    initial_gammas = np.pi / 3 * np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])  # scaled for cubic graph
    num_combinations = len(initial_betas) * len(initial_gammas) * len(depths)

    n = 20
    graph = generate_graph(n)          
    c_max = cmax_ortools_exact(graph)
    print(f"C_MAX: {c_max}")


    for zoom_idx in range(num_of_zooms):
        print(f" Zoom Level {zoom_idx + 1}")


        best_acc = -1.0
        best_params = None

            
        for initial_gamma in initial_gammas:
            for initial_beta in initial_betas:
                for depth in depths:
                    runs = []
                    num_of_runs = 3
                    for _ in range(num_of_runs):
                        val = run_qaoa(n, graph, depth, initial_gamma, initial_beta)/c_max
                        runs.append(val)
                        if val > best_acc:
                            best_acc = val
                            best_params = (initial_beta, initial_gamma, depth)
            
        best_beta, best_gamma, best_depth = best_params

        # ---- Logarithmic zoom around the best β, γ ----
        beta_min = max(best_beta * 0.5, 1e-3)
        beta_max = min(best_beta * 1.5, np.pi/2)
        gamma_min = max(best_gamma * 0.5, 1e-3)
        gamma_max = min(best_gamma * 1.5, np.pi)
        
        # refine 6×6 grid around the best region (log spacing)
        initial_betas = np.geomspace(beta_min, beta_max, 6)
        initial_gammas = np.geomspace(gamma_min, gamma_max, 6)
        depths = [best_depth]  # lock depth for subsequent zooms

        print(f"Next β range: {initial_betas}")
        print(f"Next γ range: {initial_gammas}")
        print(f"Depth locked to {best_depth}")

    
    
    
    
    
    
    
   
if __name__ == "__main__":
    main()
