# Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/virtual/2025/poster/117995) [![Website](https://img.shields.io/badge/Website-phi--rom.github.io-green)](https://phi-rom.github.io) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 

**🚧 This repository is currently being updated. Please check back soon for more information. 🚧**

This repository contains the official JAX implementation for the paper **"Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers"**, accepted at NeurIPS 2025.

$\Phi$-ROM is a novel framework for creating **Physics-informed Reduced-Order Models** for time-dependent and parameterized Partial Differential Equations (PDEs). It incorporates **differentiable PDE solvers** directly into the training loop, ensuring the learned latent dynamics align closely with the discretized governing physics. This approach enhances generalization to unseen parameters, improves long-term forecasting, and works effectively even with sparse or irregular data.

---
## Key Features ✨

* **Physics-Informed Training**: Leverages differentiable PDE solvers (like JAX-CFD, XLB, Exponax) during training to enforce physical consistency in the latent space.
* **Enhanced Generalization**: Outperforms purely data-driven ROMs and other physics-informed strategies in generalizing to new dynamics from unseen parameters and initial conditions.
* **Improved Forecasting**: Enables accurate long-term forecasting beyond the training time horizon.
* **Mesh-Free & Continuous**: Utilizes Implicit Neural Representations (INRs) and Neural ODEs for a continuous representation in both space and time, allowing flexibility with grids.
* **Sparse Data Handling**: Capable of training and recovering full solution fields even from sparse and irregular observations.
* **Data Efficiency**: Achieves better performance with less training data compared to data-driven methods.
* **Extensible Framework**: Built with JAX, Equinox, Diffrax, and Optax, making it readily adaptable to other PDE systems and differentiable solvers in JAX.

---
## Repository Structure 📂

```

├── PHIROM/          
│   ├── modules/      \# Core model components (Decoders, Dynamics Networks)
│   ├── pde/          \# PDE definitions, data loaders, and solver interfaces
│   ├── training/     \# Trainer classes (PhiROM, DINo, CROM), callbacks, evaluation metrics
│   └── utils/        \# Utility functions (experiment setup, serialization)
├── data/
│   ├── scripts/      \# Scripts to generate datasets (Burgers, Diffusion, KdV, LBM, N-S)
│   └── *.h5          \# Placeholder for dataset files
├── experiments/      \# Bash scripts for running the experiments from the paper
├── script\_*.py       \# Main executable scripts for training specific models/datasets from the paper
├── requirements.txt  \# Python package dependencies
└── README.md         \# This file

````

---

## Installation 🔧

1.  **Clone the repository:**
    ````bash
    git clone https://github.com/phi-rom/PhiROM.git
    cd PhiROM
    ````
    
2.  **Create a virtual environment (recommended):**
    ````bash
    python -m venv phirom_env
    source phirom_env/bin/activate 
    ````
3.  **Install dependencies:**
    Phi-ROM requires JAX with GPU support. The requirements specify JAX with CUDA 12 support.
    ````bash
    pip install -r requirements.txt
    ````

    3.1 **Install Torch**
    You need to separately install the CPU version of [PyTorch](https://pytorch.org/) (used for data loading):
    ````bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ````

    3.2 **Install XLB**
    To run the Lattice Boltzmann experiments, install the latest [XLB](https://github.com/Autodesk/XLB) library:
    ````bash
    pip install git+https://github.com/Autodesk/XLB.git
    ````
---

<!-- ## Usage 🚀

### 1. Data Generation

Scripts for generating the datasets used in the paper are provided in the `data/scripts/` directory. For example, to generate the Navier-Stokes turbulence dataset:
````bash
python data/scripts/turbulence.py
````
    
Generated `.h5` files should be placed in the `data/` directory.

### 2\. Training Models

Training scripts for various configurations (Φ-ROM, DINo, PINN-ROM, CROM) and datasets (Burgers, Diffusion, KdV, LBM, N-S) are located in the `experiments/` directory.

You can run a specific training experiment using its corresponding shell script. For example, to train Φ-ROM on the Navier-Stokes turbulence dataset:

````bash
bash experiments/train_turbulence_PHIROM.sh
````

Alternatively, you can run the main Python scripts directly, passing hyperparameters as command-line arguments. For example:

````bash
python script_turbulence.py --latent_dim=100 --width=80 --node_width=512 --epochs=24000 --dataset='ns_turbulence_new_64x64_ins=5' --loss_lambda=0.5 --gamma=0.1 --evolve_start=100 ...
````

Refer to the individual `script_*.py` files and the `experiments/*.sh` scripts for available arguments and configurations. Training checkpoints and results will be saved in the `NODE_experiments/` or `CROM_experiments/` directory, organized by dataset and hyperparameters.

### 3\. Evaluation and Inference

The `PHIROM/training/evaluation.py` and `PHIROM/modules/inference.py` modules contain functions for evaluating trained models and performing inference (forecasting). The training scripts utilize evaluation callbacks (`PHIROM/training/callbacks.py`) to monitor performance during training.

----- -->

## Citation 📝

If you find this work useful, please cite our NeurIPS 2025 paper:

````bibtex
@inproceedings{hosseini2025phirom,
  title={{Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers}},
  author={Hosseini Dashtbayaz, Nima and Salehipour, Hesam and Butscher, Adrian and Morris, Nigel},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  volume={39}
}
````
