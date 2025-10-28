# QAOA Implementation with Qiskit

## Overview
This repository contains an implementation of the Quantum Approximate Optimisation Algorithm (QAOA) using IBM Qiskit.
The project demonstrates how QAOA can be applied to combinatorial optimisation problems, focusing on the Max-Cut problem as a case study.
It is designed to be both educational and extendable for researchers or engineers exploring hybrid quantum-classical algorithms.

---

## Key Features
- End-to-end QAOA pipeline using Qiskit
- Example implementation for the Max-Cut problem on a small graph
- Execution on both simulator (Aer) and real IBM Quantum hardware
- Integration with classical optimisers (e.g COBYLA, SPSA) 
- Visualisation of results and circuit structure 
- Modular code that can be adapted for other optimisation problems


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

## Example Results

For a 4-node graph, QAOA finds the optimal Max-Cut configuration consistent with the classical solution.   
The code also reports the cut value and the bitstring corresponding to the best measurement outcome.

# To-do (ADD EXAMPLE PLOT HERE)


## Author
**Jamie Holdcroft**
GitHub: [https://github.com/JamieHoldcroft](https://github.com/JamieHoldcroft)
