# Project Overview

This repository implements PraNet-based polyp segmentation and introduces an
Hippocampal Neural Network (HNN) continual-learning mechanism inspired by a
Mott-VO2 hippocampus circuit. The code and algorithms were developed together
with the hardware concept so that temporal device primitives (time-constants,
selective decay, and continuous integration of inputs) are reflected in data
flow, stored artifacts, and the hippocumpal cognitve map aggregation. 
Alongside model implementations, the repository provides training, testing 
and evaluation scripts plus dataloaders to reproduce the experiments that 
explore this hardware–software approach.

---

## High-level structure

- `*.py` (root): various training and utility scripts (examples: `mytrain_cl.py`, `mytest_cl.py`, `myeval_cl.py`).
- `lib/`: model definitions and backbones (PraNet variants, Res2Net, UNet, ResNet implementations).
- `utils/`: helper utilities and dataloaders. There are multiple dataloader variants to support different dataset organisations and continual-learning experiments.

Files of interest (non-exhaustive):
- `mytrain_cl.py`: training script. This repository contains `mytrain_cl.py` as the primary training entry; it implements EWC/HNN style logic.
- `mytest_cl.py`: inference/test script that handles multi-head outputs and saves maps/overlays.

- `lib/PraNet_Res2Net_cl.py`: PraNet implementation adapted to support multiple task-specific reverse-attention heads for continual learning.

---


## HNN (Hippocampal Neural Network) — primary contribution

The main methodological contribution in this repository is the HNN-style
aggregation: a hippocampal-inspired mechanism that decays and aggregates per-
task cognitive maps so the model retains an ordered, importance-weighted
memory of prior tasks. HNN is implemented together with an regularization penalty to
consolidate important parameters while allowing task-specific heads to remain
plastic. Key ideas and implementation points:

- After finishing training on a task, the code computes diagonal
  approximations of the Fisher information (squared gradients averaged
  over the task's data) and stores them in a `fisher_dict` keyed by task.
- When training subsequent tasks, an aggregated importance map called
  `cognitve_map_HNN` is constructed by decaying and summing previous Fisher
  tensors. The adjustment is a function of learning time (e.g., `decay = t - key`)—
  older tasks receive stronger attenuation. This models a hippocampal-like
  episodic memory with fading influence over time (order + importance).
- The computed Fisher (either the immediate `fisher_information` or the
  aggregated `cognitive_map_HNN`) is used to penalize deviations of 
  important parameters from their stored values. 

This combination (per-task Fisher storage + learning aggregation +
selective adjustment) is intended to capture the order and importance of past
learning episodes and to provide a practical consolidation mechanism for
continual segmentation tasks.


## Example commands

Train (example, adjust options):

```bash
python mytrain_cl.py --epoch 40 --lr 1e-4 --batchsize 8 --trainsize 256 --approches HNN
```

Test / inference (example):

```bash
python mytest_cl.py --weights_dir path/to/weights_dir --approches HNN
```

Evaluate (example):

```bash
python myeval_cl.py --weights_dir path/to/weights_dir --approches HNN
```


## Environment and reproducibility (exact setup used)

Below are concrete environment details derived from the conda environment
you provided. Use these instructions to reproduce the runtime environment
used to develop and test this repository.

Key versions used (from your `conda list`):
- Python: 3.10.11
- PyTorch: 1.13.0 (CUDA-enabled build: cu116, built with CUDA 11.6 runtime)
- torchvision: 0.14.0
- torchaudio: 0.13.0
- CUDA toolkit (system/conda meta): 11.6 / cudatoolkit 11.8 present
- numpy==1.23.5
- scipy==1.10.1
- scikit-image==0.19.3
- pandas==1.5.3
- pillow==9.4.0
- opencv-python (pypi) 3.4.16.59
