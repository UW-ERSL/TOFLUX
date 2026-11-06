<h1 align="center">
  <a href="https://link.springer.com/article/10.1007/s00158-026-04252-7">
    TOFLUX: A Differentiable Topology Optimization Framework for Multiphysics Fluidic Problems
  </a>
</h1>

<p align="center">
  <a href="https://sites.google.com/view/rahulkp/home">Rahul Kumar Padhy</a>,
  <a href="https://scholar.google.com/citations?hl=en&user=hqoL27AAAAAJ&view_op=list_works&sortby=pubdate">Krishnan Suresh</a>,
  <a href="https://scholar.google.com/citations?user=9wCpPIkAAAAJ&hl=en">Aaditya Chandrasekhar</a>
</p>


## Abstract

Topology Optimization (TO) holds the promise of designing next-generation compact and efficient fluidic devices. However, the inherent complexity of fluid-based TO systems, characterized by multiphysics nonlinear interactions, poses substantial barriers to entry for researchers.

Beyond the inherent intricacies of forward simulation models, design optimization is further complicated by the difficulty of computing sensitivities, i.e., gradients. Manual derivation and implementation of sensitivities are often laborious and prone to errors, particularly for non-trivial objectives, constraints, and material models. An alternative solution is automatic differentiation (AD). Although AD has been previously demonstrated for simpler TO problems, extending its use to complex nonlinear multiphysics systems, specifically in fluidic optimization, is key to reducing the entry barrier.

To this end, we introduce TOFLUX, a TO framework for fluid devices leveraging the JAX library for high-performance automatic differentiation. The flexibility afforded by AD enables the rapid exploration and evaluation of various objectives and constraints. We illustrate this capability through challenging examples encompassing thermo-fluidic coupling, fluid-structure interaction, and non-Newtonian flows. Additionally, we demonstrate the seamless integration of our framework with neural networks and machine learning methodologies, enabling modern approaches to scientific computing. Ultimately, the framework aims to provide a foundational resource to accelerate research and innovation in fluid-based TO.

## Codebase Organization

The project is organized into the following main directories:

*   [`toflux/src/`](toflux/src): Contains the core source code for the framework.
*   [`toflux/experiments/`](toflux/experiments): Contains Jupyter notebooks that demonstrate various applications of the framework.
*   [`toflux/brep/`](toflux/brep): Contains geometry definitions used in the experiments.

## Experiments

The [`toflux/experiments/`](toflux/experiments) directory contains notebooks that serve as examples for using the framework.

*   **Structural TO**: [`toflux/experiments/structural/`](toflux/experiments/structural)
    *   `solve.ipynb`: A simple forward analysis of a L-beam to demonstrate the basic structural mechanics solver.
    *   `topopt.ipynb`: The classic compliance minimization problem for an L-beam.

*   **Fluidic TO**: [`toflux/experiments/fluid/`](toflux/experiments/fluid)
    *   `solve.ipynb`: Forward simulation of Navier-Stokes flow in a channel.
    *   `topopt_dissip_power.ipynb`: Optimizes a double pipe design to minimize the total power dissipated by the fluid.
    *   `topopt_drag.ipynb`: Designs a airfoil structure in an exterior flow to minimize drag.
    *   `topopt_flow_reverse.ipynb`: Optimizes a device to reverse the direction of fluid flow.

*   **Fluid-Structure Interaction (FSI)**: [`toflux/experiments/fsi/`](toflux/experiments/fsi)
    *   `solve.ipynb`: Demonstrates the one-way coupled FSI solver, where fluid pressure acts on a compliant structure.
    *   `topopt.ipynb`: An optimization problem where the goal is to design a wall that has maximum stiffness under fluid loading.

*   **Conjugate Heat Transfer (CHT)**: [`toflux/experiments/conjugate_heat/`](toflux/experiments/conjugate_heat)
    *   `solve.ipynb`: Simulates heat transfer in a channel flow between cold incoming fluid and heated walls.
    *   `topopt.ipynb`: Optimizes the topology of a channel to maximize recoverable thermal power and minimize fluid dissipation.

*   **Non-Newtonian Flows**: [`toflux/experiments/non_newtonian/`](toflux/experiments/non_newtonian)
    *   `topopt.ipynb`: Shows the flexibility of the framework by implementing and optimizing a device with a fluid that follows a non-Newtonian Carreau Yasuda viscosity model.

*   **Neural Network Integration**: [`toflux/experiments/fluid/tounn/`](toflux/experiments/fluid/tounn)
    *   `topopt.ipynb`: This example demonstrates how TOFLUX can be integrated with machine learning libraries. A neural network (defined in `network.py`) is used to parameterize the design space, and the network weights are optimized to produce a final design.

## Citation

```bibtex
@article{padhy2026toflux,
  title={Toflux: a differentiable topology optimization framework for multiphysics fluidic problems},
  author={Padhy, Rahul Kumar and Suresh, Krishnan and Chandrasekhar, Aaditya},
  journal={Structural and Multidisciplinary Optimization},
  volume={69},
  number={3},
  pages={56},
  year={2026},
  publisher={Springer}
}
