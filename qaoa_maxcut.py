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

# def generate_petersen_graph() -> rx.PyGraph:
#     g = rx.PyGraph()
#     g.add_nodes_from(range(10))
#     edges = [
#         # outer 5-cycle
#         (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
#         # inner 5-point star (5-7-9-6-8-5)
#         (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
#         # spokes
#         (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
#     ]
#     g.add_edges_from([(u, v, 1.0) for (u, v) in edges])
#     return g

# graph = generate_petersen_graph()
# n = len(list(graph.nodes()))  

# plt.figure(figsize=(5, 5))
# draw_graph(graph, node_size=600, with_labels=True)
# plt.title("Petersen Graph")
# plt.axis("off")
# plt.show()

# -----------------------------
# Part 1. Small-Scale QAOA
# -----------------------------

# Step 1: Build small graph (5 nodes)
n = 5
graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [
    (0, 1, 1.0),
    (0, 2, 1.0),
    (0, 4, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 4, 1.0),
]
graph.add_edges_from(edge_list)

draw_graph(graph, node_size=600, with_labels=True)

# Helper: build Max-Cut Hamiltonian
def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)
print("Cost Function Hamiltonian:\n", cost_hamiltonian)

# Step 2: Build QAOA circuit
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()
circuit.draw("mpl")

# Step 3: Optimize circuit for backend
backend = AerSimulator()
print(f"Using backend: {backend}")

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
candidate_circuit = pm.run(circuit)
candidate_circuit.draw("mpl", fold=False, idle_wires=False)

# Step 4: Define cost function
initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
objective_func_vals = []

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    result = job.result()[0]
    cost = result.data.evs
    objective_func_vals.append(cost)
    return cost

# Run optimization locally
estimator = Estimator(mode=backend)
result = minimize(
    cost_func_estimator,
    init_params,
    args=(candidate_circuit, cost_hamiltonian, estimator),
    method="COBYLA",
    tol=1e-2,
)
print("\nOptimization result:\n", result)

# Step 5: Plot optimization convergence
plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Step 6: Execute final circuit
optimized_circuit = candidate_circuit.assign_parameters(result.x)
sampler = Sampler(mode=backend)
job = sampler.run([(optimized_circuit,)], shots=10_000)
counts_int = job.result()[0].data.meas.get_int_counts()
shots = sum(counts_int.values())
final_distribution_int = {k: v / shots for k, v in counts_int.items()}
print("\nFinal distribution:\n", final_distribution_int)

# Step 7: Post-process results
def to_bitstring(integer, num_bits):
    return [int(digit) for digit in np.binary_repr(integer, width=num_bits)]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, len(graph))
most_likely_bitstring.reverse()
print("\nMost likely bitstring:", most_likely_bitstring)

# Step 8: Plot result distribution
matplotlib.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(11, 6))
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")
ax.bar(list(final_distribution_int.keys()), list(final_distribution_int.values()), color="tab:grey")
plt.show()

# Step 9: Visualize the best cut
def plot_result(G, x):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos = rx.spring_layout(G)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=200, alpha=0.8, pos=pos)

plot_result(graph, most_likely_bitstring)

# Step 10: Evaluate the cut value
def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes()))
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))

cut_value = evaluate_sample(most_likely_bitstring, graph)
print("\nThe value of the cut is:", cut_value)
