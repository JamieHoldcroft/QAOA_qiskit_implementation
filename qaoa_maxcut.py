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

# Grid search initial param values
depths = [1, 2, 3, 4, 5]
beta_grid = np.array([np.pi/12, np.pi/8, np.pi/6, np.pi/4, 3*np.pi/8, np.pi/2])
gamma_grid = np.pi / 3 * np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])  # scaled for cubic graph



def generate_graph(n):
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    edge_list = [
    # outer 5 cycle
    (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 0, 1.0),
    # inner 5 point star
    (5, 7, 1.0), (7, 9, 1.0), (9, 6, 1.0), (6, 8, 1.0), (8, 5, 1.0),
    # spokes
    (0, 5, 1.0), (1, 6, 1.0), (2, 7, 1.0), (3, 8, 1.0), (4, 9, 1.0),
    ]
    graph.add_edges_from(edge_list)
    draw_graph(graph, node_size=600, with_labels=True)
    return graph

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
    print("Cost Function Hamiltonian:\n", cost_hamiltonian)

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


# -----------------------------
# Main flow (same behavior, just function calls)
# -----------------------------

def main():
    # Problem
    n = 10
    graph = generate_graph(n)
    # c_max = cmax_ortools_exact(graph)
    print("C_MAX")
    # print(c_max)

    # Build cost + circuit
    cost_hamiltonian, circuit = build_cost_and_circuit(graph, n, reps=2)

    # Backend
    backend = AerSimulator()
    print(f"Using backend: {backend}")

    # Transpile for backend
    candidate_circuit = optimize_for_backend(circuit, backend)

    # Initial params (same as yours)
    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

    # Optimize
    result = run_optimization(candidate_circuit, cost_hamiltonian, backend, init_params)
    print("\nOptimization result:\n", result)

    # Plot convergence
    plot_convergence(objective_func_vals)

    # Sample final circuit
    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    final_distribution_int = sample_distribution(optimized_circuit, backend, shots=10_000)
    print("\nFinal distribution:\n", final_distribution_int)

    # Most likely bitstring
    most_likely_bitstring = most_likely_bitstring_from_dist(final_distribution_int, len(graph))
    print("\nMost likely bitstring:", most_likely_bitstring)
    

if __name__ == "__main__":
    main()
