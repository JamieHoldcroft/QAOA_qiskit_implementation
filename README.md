# QAOA Implementation with Qiskit

## Overview
This repository reimplements the Quantum Approximate Optimisation Algorithm (QAOA) using Qiskit and the official IBM Qiskit QAOA tutorial.
The goal of the project is to understand every step of the Quantum Approximate Optimization Algorithm (QAOA) and to explore modifications and variations of the QAOA as outlined in the personal extensions section.

The tutorial results are reproduced on local simulators, with the aim of ensuring a deep understanding of how each Qiskit component interacts (e.g SparsePauliOp, QAOAAnsatz, EstimatorV2, and SamplerV2).

### Motivation
This project serves as a practical learning exercise to:
- Strengthen understanding of quantum optimization workflows.
- Gain experience using Qiskit Runtime primitives and local simulators.
- Build a foundation for future quantum algorithm research and hardware experimentation.

## Personal Extensions
- Experimenting with custom graph structures (random, dense, and weighted graphs).

- Running QAOA on IBM Quantum hardware via real backends and comparing results with simulators.

- Adding noise models and error mitigation to explore realistic performance.

- Extending to larger graphs (e.g., 20–100 qubits) to study scalability.

- Implementing custom optimisation loops and visualizing convergence metrics.

## Key Features
- End-to-end QAOA pipeline using Qiskit
- Example implementation for the Max-Cut problem on a small graph
- Execution on both simulator (Aer) and real IBM Quantum hardware
- Integration with classical optimisers (e.g COBYLA, SPSA) 
- Visualisation of results and circuit structure 
- Modular code that can be adapted for other optimisation problems
- Understanding of how variations to the QAOA alters performance

---

## Installation
Clone the repository and install the required dependencies.

git clone https://github.com/JamieHoldcroft/QAOA_qiskit_implementation   
cd QAOA_qiskit_implementation   
python -m venv .venv
### Windows
.venv\Scripts\activate
### macOS/Linux
source .venv/bin/activate   
pip install -r requirements.txt

---

## Running on IBM Quantum Hardware (Optional)

To execute on a real backend:
1. Create an IBM Quantum account at quantum.ibm.com
2. Retrieve your API token and save it in your environment:   
    - from qiskit_ibm_runtime import QiskitRuntimeService QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_API_KEY")
3. Update the backend in the script:   
    - backend = QiskitRuntimeService().backend("ibm_heron_r3")

---

## Author
**Jamie Holdcroft**

IBM QAOO Qiskit Tutorial: [https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)

GitHub: [https://github.com/JamieHoldcroft](https://github.com/JamieHoldcroft)
