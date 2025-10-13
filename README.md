<h1 align="center">
  <a href="https://arxiv.org/abs/2508.17564">
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

## Installation

We strongly recommend using a virtual environment to manage project dependencies and avoid conflicts with other Python projects on your system.

### Using Poetry

[Poetry](https://python-poetry.org/) is a modern tool for dependency management and packaging in Python.

1.  **Install Poetry:** If you don't have Poetry, follow the [official installation guide](https://python-poetry.org/docs/#installation).
2.  **Clone the repository:**
    ```sh
    git clone git@github.com:UW-ERSL/TOFLUX.git
    cd TOFLUX
    ```
3.  **Install dependencies:** This command will create a new virtual environment if one doesn't exist and install all the required packages.
    ```sh
    poetry install
    ```
4.  **Activate the virtual environment:**
    ```sh
    poetry shell
    ```

### Using pip and venv

You can also use `pip` with Python's built-in `venv` module.

1.  **Clone the repository:**
    ```sh
    git clone git@github.com:UW-ERSL/TOFLUX.git
    cd TOFLUX
    ```
2.  **Create a virtual environment:**
    ```sh
    python3 -m venv .venv
    ```
3.  **Activate the virtual environment:**
    *   On macOS and Linux:
        ```sh
        source .venv/bin/activate
        ```
    *   On Windows:
        ```sh
        .\.venv\Scripts\activate
        ```
4.  **Install the project in editable mode:**
    ```sh
    pip install -e .
    ```
    
### Using Conda

1.  **Install Conda:** If you don't have Conda, follow the [official installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install).
2.  **Clone the repository:**
    ```sh
    git clone git@github.com:UW-ERSL/TOFLUX.git
    cd TOFLUX
    ```
3.  **Install dependencies:** Create the environment and install dependencies.
    ```sh
    conda create -n toflux -c conda-forge -y python=3.11 jax=0.6.0 jaxlib=0.6.0 numpy=2.2.6 scipy=1.15.2 matplotlib "shapely>=2.0,<3.0" pandas pyyaml chex=0.1.88 pyamg=5.2.1
    ```
4.  **Activate the environment:**
    ```sh
    conda activate toflux
    ```
### A Note on JAX Installation

This project uses [JAX](https://github.com/google/jax) for automatic differentiation and hardware acceleration. The default installation will use the CPU. If you have a GPU (NVIDIA) or TPU (Google), you can install a version of JAX that supports it for significant performance improvements. Please follow the [official JAX installation instructions](https://github.com/google/jax#installation) to install JAX with the appropriate CUDA or ROCm drivers.

## Codebase Organization

The project is organized into the following main directories:

*   [`toflux/src/`](toflux/src): Contains the core, reusable source code for the framework, including modules for finite element analysis ([`fe_fluid.py`](toflux/src/fe_fluid.py), [`fe_struct.py`](toflux/src/fe_struct.py), [`fe_thermal.py`](toflux/src/fe_thermal.py)), meshing ([`mesher.py`](toflux/src/mesher.py)), material models ([`material.py`](toflux/src/material.py)), boundary conditions ([`bc.py`](toflux/src/bc.py)), solvers ([`solver.py`](toflux/src/solver.py)), and optimization ([`mma.py`](toflux/src/mma.py)).
*   [`toflux/experiments/`](toflux/experiments): Contains Jupyter notebooks that demonstrate various applications of the framework. Each sub-directory corresponds to a specific multiphysics problem.
*   [`toflux/brep/`](toflux/brep): Contains geometry definitions in `.json` format used in the experiments.
*   [`toflux/figures/`](toflux/figures): Contains images and figures generated by the experiments.

## Experiments

The [`toflux/experiments/`](toflux/experiments) directory contains a series of Jupyter notebooks that serve as examples and starting points for using the framework. Each notebook is self-contained and demonstrates a specific physics problem or optimization setup.

*   **Structural TO**: [`toflux/experiments/structural/`](toflux/experiments/structural)
    *   `solve.ipynb`: A simple forward analysis of a L-beam to demonstrate the basic structural mechanics solver.
    *   `topopt.ipynb`: The classic compliance minimization problem for an L-beam. This is a good starting point for understanding the basic TO workflow in TOFLUX.

*   **Fluidic TO**: [`toflux/experiments/fluid/`](toflux/experiments/fluid)
    *   `solve.ipynb`: Forward simulation of Navier-Stokes flow in a channel.
    *   `topopt_dissip_power.ipynb`: Optimizes a double pipe design to minimize the total power dissipated by the fluid.
    *   `topopt_drag.ipynb`: Designs a airfoil structure in an exterior flow to minimize drag.
    *   `topopt_flow_reverse.ipynb`: A more advanced example that optimizes a device to reverse the direction of fluid flow, a common benchmark in fluid TO.

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

## Requirements

The project relies on several key libraries. The major ones are:
*   `jax`
*   `numpy`
*   `matplotlib`
*   `optax`
*   `flax`
*   `pyyaml`
*   `scipy`

Please see the `pyproject.toml` file for a complete list of dependencies.

## Related Projects

* [PhiFlow](https://tum-pbs.github.io/PhiFlow/):
Differentiable PDE/physics toolkit (fluids included) in Python with NumPy/PyTorch/JAX/TensorFlow backends, built for optimization and ML workflows.

* [Diff-FlowFSI](https://arxiv.org/abs/2505.23940): A GPU-Optimized Differentiable CFD Platform for High-Fidelity Turbulence and FSI Simulations — Fully differentiable, GPU-accelerated CFD platform targeting turbulence and FSI.

## Citation

If you use TOFLUX in your research, please consider citing it.

```bibtex
@article{padhy2025toflux,
  title={TOFLUX: A Differentiable Topology Optimization Framework for Multiphysics Fluidic Problems},
  author={Padhy, Rahul Kumar and Suresh, Krishnan and Chandrasekhar, Aaditya},
  journal={arXiv preprint arXiv:2508.17564},
  year={2025}
}
