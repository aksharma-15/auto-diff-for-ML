# auto-diff-for-ML
A beginners guide on auto differentiation using PyTorch autograd

# Deep-Dive into Automatic Differentiation (Auto-Diff) for Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An advanced, foundational repository demonstrating the mechanics of **Automatic Differentiation (Auto-Diff)**, **Computational Graphs**, and **Gradient Verification** from scratch using **PyTorch Autograd**. 

This project is tailored to showcase core production-level competencies in deep learning framework internals, mathematical optimization, and rigorous debugging techniques (Gradient Checking) necessary for engineering robust machine learning systems.

---

## 📌 Technical Overview & Objectives

In modern deep learning, models contain millions or billions of parameters. Computing gradients manually via analytical calculus is slow, highly error-prone, and practically impossible for complex architectures. This repository contains a comprehensive breakdown of how modern ML frameworks handle optimization seamlessly.

### Core Architecture Covered:
1. **Dynamic Computational Graphs (DCG):** How PyTorch builds and executes reverse-mode automatic differentiation on-the-fly.
2. **Framework Internals:** Managing `requires_grad`, tracking tensor lineages, and inspecting the `.grad` accumulator.
3. **Mathematical Verification:** Implementing numerical gradient approximations using **Finite Differences** to cross-validate framework-generated analytical gradients.
4. **Optimization Foundations:** Executing clean, lower-level gradient descent loops, managing graph tracking memory via context managers (`torch.no_grad()`), and preventing gradient accumulation bugs (`.zero_()`).

---

## 🧮 Mathematical Foundations

### 1. Reverse-Mode Automatic Differentiation
PyTorch utilizes a **Dynamic Computational Graph** to execute reverse-mode automatic differentiation. During the forward pass, operations are evaluated sequentially while a directed acyclic graph (DAG) of execution is built in memory. Nodes represent tensors, and edges represent primitive mathematical operations.

When `.backward()` is invoked, the framework applies the **Calculus Chain Rule** from the output (typically the loss scalar) backwards to the leaf nodes (parameters):

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}$$

This method is highly optimized ($O(1)$ scaling cost relative to the number of parameters) for functions with many inputs and a single scalar output, making it the bedrock of neural network training.

### 2. Finite Differences (Gradient Checking)
To guarantee that the analytical gradients computed by `autograd` are absolutely correct (vital when writing custom autograd functions or custom loss layers), we apply the **Two-Sided Finite Differences formula**:

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Where $\epsilon$ is an ultra-small perturbation constant (e.g., $10^{-5}$). Comparing the analytical gradient ($G_a$) and numerical gradient ($G_n$) ensures mathematical correctness if their relative error is within strict tolerances:

$$\text{Relative Error} = \frac{|G_a - G_n|}{|G_a| + |G_n|} < 10^{-7}$$

---

## ⚙️ Hardware-Agnostic Installation Guide

PyTorch must be compiled against specific hardware architectures to optimize memory transfers and parallel tensor execution. Execute the command matching your local environment below:

```bash
# Standard CPU-Only Architecture
pip install torch torchvision torchaudio

# Apple Silicon Native Acceleration (M1 / M2 / M3 Metal Performance Shaders)
pip install torch torchvision

# NVIDIA GPU Acceleration via Unified Architecture (CUDA 12.6 Compilation)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

## **Brief introduction of the Concepts in the jupyter playbook**
### Concept 1: Tensors & Multi-Dimensional Linearity 
Demonstrates low-level structural data execution using multi-dimensional matrices representing input dimensions, weights ($W$), and biases ($b$) for linear tracking models ($y = Wx + b$).

```import torch

# Multi-dimensional structural inputs & hyper-parameters
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
b = torch.tensor([0.1, 0.2])

# Executing structural matrix multiplication mapping to lower-dimensional space
y = torch.matmul(x, w) + b
print(f"Prediction Vector: {y}") # Returns tensor([11.1000, 14.2000])
```

### Concept 2: Reverse-Mode Autograd in Action
Tracking forward operations dynamically and executing automatic backpropagation.

```
# Initialize leaf nodes with graph tracking enabled
a = torch.tensor(5.0, requires_grad=True)
b = a**3 + 4*a + 2

# Execute backpropagation through the computational graph
b.backward()

# Analytic Derivative: db/da = 3a^2 + 4 -> 3(25) + 4 = 79
print(f"Analytical Gradient: {a.grad}") # Returns tensor(79.)
```

### Concept 3 & 4: Analytical vs. Numerical Gradient Verifier
A comprehensive comparative module built to safeguard architectures against silent gradient corruption.

```
import numpy as np

def f(x):
    return x**3 + 2*x

# Analytical tracking using exact calculus formulas (PyTorch Autograd)
x_analytic = torch.tensor(2.0, requires_grad=True)
y_analytic = f(x_analytic)
y_analytic.backward()
g_analytic = x_analytic.grad.item()

# Numerical approximation via Central Finite Difference Engine
x_val = 2.0
h = 1e-5
g_numeric = (f(x_val + h) - f(x_val - h)) / (2 * h)

print(f"Analytical Exact Value: {g_analytic}") # 14.0
print(f"Numerical Approximation: {g_numeric}") # 14.000000000180533
```


## **Getting started**
1. Prerequisites
Ensure you have Python 3.10+ installed. You can check your local version by running:
```python --version```

2. Clone the Repository
Grab the codebase by cloning the repository and navigating into the project root:
```
git clone https://github.com/aksharma-15/auto-diff-for-ML.git
cd auto-diff-for-ML
```
3. Install Dependencies
Install the project ecosystem. Select the command that corresponds to your hardware setup to ensure proper optimization and compute acceleration:
```
# Option A: Standard CPU-Only Setup
pip install torch torchvision torchaudio notebook numpy

# Option B: Apple Silicon Native Acceleration (M1 / M2 / M3 MPS)
pip install torch torchvision notebook numpy

# Option C: NVIDIA GPU Acceleration (CUDA 12.6 Compliant)
pip install torch torchvision notebook numpy --index-url https://download.pytorch.org/whl/cu126
```
4. Run the Interactive Pipeline
Launch the Jupyter instance to step through the core modules, computational graph visuals, and gradient checks:
```jupyter notebook auto_diff_for_ML.ipynb```


Connect with me on LinkedIn - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abhay-kumar-sharma-a22a94171)
