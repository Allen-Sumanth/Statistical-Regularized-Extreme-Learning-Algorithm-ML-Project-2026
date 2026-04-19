# Implementing Regularised Extreme Learning Machines for Multicollinear Data

## Introduction

Machine learning models frequently struggle when input features exhibit high correlation, a phenomenon known as multicollinearity. This issue causes standard learning algorithms to become unstable, leading to wildly inaccurate predictive models. This repository contains a complete testing framework built from scratch to replicate and evaluate the findings of a recent paper proposing novel algorithms designed to address this fundamental flaw.

This project is a direct implementation and replication study based on the research paper: **"Novel Statistical Regularized Extreme Learning Algorithm to Address the Multicollinearity in Machine Learning"** by H. Yildirim (IEEE Access, vol. 12, pp. 102355-102367, 2024, doi: 10.1109/ACCESS.2024.3432490). We evaluate the proposed Two-Parameter Extreme Learning Machine (TP1-ELM and TP2-ELM) methods against existing industry standards across multiple benchmark datasets.

## Project Authors

* **Allen Sumanth A** (108123012)
* **Adithya S** (108123006)

## The Multicollinearity Problem in ELMs

The Universal Approximation Theorem dictates that a single-layer feed-forward network can approximate any mathematical function. Extreme Learning Machines (ELMs) leverage this by randomly assigning weights to the hidden layer, completely skipping the iterative backpropagation process. Training an ELM is reduced to a single, lightning-fast step: calculating the output weights using matrix inversion.

The core relationship is defined by the equation $H\beta = Y$. Here, $H$ represents the output of the hidden layer, $\beta$ is the matrix of final connection weights we are trying to solve for, and $Y$ represents the correct answers (the target variable).

Because $H$ is rarely a perfect square matrix, standard inversion is impossible. ELMs solve this by using the Moore-Penrose generalized inverse ($H^\dagger$), calculated as:

$\beta = (H^T H)^{-1} H^T Y$

The critical point of failure occurs at the matrix $(H^T H)$. When the input data contains highly correlated features, the columns of the $H$ matrix become linearly dependent. This causes the determinant of $(H^T H)$ to approach zero, making the matrix ill-conditioned and nearly singular. Inverting such a matrix causes the weights in $\beta$ to explode, leading to massive instability and failure in the model.

## Implemented Algorithms

This framework implements the standard ELM and several regularised variants designed to penalise large weights and stabilise the matrix inversion process:

* **Base ELM:** The foundational model using the standard Moore-Penrose pseudoinverse.
* **Ridge-ELM:** Incorporates $L_2$ regularisation, adding a penalty parameter $k$ to the diagonal of the matrix.
* **Liu-ELM:** Utilises a parameter $d$ to shrink the weights, offering an alternative to standard Ridge regression.
* **TP1-ELM (Two-Parameter 1):** The first novel algorithm from the replicated paper, combining both $k$ and $d$ parameters to simultaneously address multicollinearity and improve model fit.
* **TP2-ELM (Two-Parameter 2):** A variant of the Two-Parameter model with an alternative mathematical formulation for calculating the output weights.

## Experimental Framework and Datasets

The testing framework fetches, standardises, and processes seven tabular regression datasets specified in the original paper. We implemented a Variance Inflation Factor (VIF) diagnostic function to empirically demonstrate the presence of multicollinearity prior to training.

The datasets evaluated are:

1. Auto Price (OpenML)
2. Boston Housing (OpenML)
3. Fish Toxicity (UCI)
4. Forest Fires (OpenML)
5. Machine CPU (OpenML)
6. Servo (UCI)
7. Slump/Strikes (OpenML)

Models are evaluated using 5-fold cross-validation, with performance measured via Root Mean Square Error (RMSE). Hyperparameter tuning is conducted across predefined grids for the regularisation parameters $k$ and $d$, while maintaining a fixed hidden layer size.

## Repository Structure and Usage

* `Accelerated_ELM_Pipeline.ipynb`: The primary Jupyter Notebook containing the complete end-to-end pipeline. This includes data fetching, VIF diagnostics, class definitions for all ELM variants, cross-validation loops, and plotting functions.
* `Project Report.pdf`: The detailed academic report documenting the theoretical background, mathematical derivations, and analysis of our experimental results.

### Dependencies

The pipeline requires Python 3.8+ and the following libraries:

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `charset-normalizer`

To run the pipeline, simply execute the cells sequentially in `Accelerated_ELM_Pipeline.ipynb`.
