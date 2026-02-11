# Evaluation of Data Efficiency in a Hybrid CNN-VQC Architecture for Brain Tumor Diagnosis in Magnetic Resonance Imaging

**Author**: Frank Edward Daza Gonzalez  
**Program**: Master in Artificial Intelligence and Data Science  
**Institution**: Universidad Autónoma de Occidente, Cali  
**Year**: 2025

---

## 📖 Introduction

Diagnostic imaging is a cornerstone of modern medicine, with Magnetic Resonance Imaging (MRI) being the gold standard for brain tumor detection. While Deep Learning (DL), specifically Convolutional Neural Networks (CNNs), has revolutionized this field, it faces a critical challenge: the **"Data Paradox"**. State-of-the-Art (SOTA) models require uniform massive datasets to avoid overfitting, but high-quality labeled medical data is intrinsically scarce and expensive to obtain (*Small Data*).

This project investigates **Quantum Machine Learning (QML)** as a promising frontier to address this limitation. It proposes a **Hybrid Architecture** integrating a classical CNN as a feature extractor with a **Variational Quantum Classifer (VQC)**. The core hypothesis is that the high expressibility of quantum circuits in high-dimensional Hilbert spaces could allow for better generalization with fewer training data compared to purely classical models.

## ❓ Problem Statement

The dominant trend in Deep Learning is massive scaling (e.g., ResNet, EfficientNet), which leads to models with millions of parameters. When trained on small medical datasets (hundreds or few thousands of images), these models are prone to **overfitting**.

**Research Question:**  
*Can a hybrid quantum-classical architecture, based on Transfer Learning, demonstrate superior generalization capacity compared to state-of-the-art classical architectures (like EfficientNet-B0) in limited-data training scenarios, while maintaining competitive accuracy metrics?*

## 🎯 Objectives

### General Objective
To evaluate the viability and efficiency of a hybrid quantum-classical architecture for multi-class brain tumor classification using MRI, comparing its performance against classical architectures in data-scarcity scenarios.

### Specific Objectives
1.  **Develop a Hybrid Architecture**: Integrate a pre-trained classical neural network (feature extractor) with a Variational Quantum Classifier (VQC) optimized using *Angle Embedding* and *Strongly Entangling Layers*.
2.  **Train Classical Baselines**: Train reference models (EfficientNet, ResNet) with the selected database.
3.  **Comparative Evaluation**: Analyze the computational efficiency of the proposal in terms of convergence time, number of trainable parameters, and simulation resource usage, comparing them with traditional deep learning models.

## 🛠️ Methodology

The project follows a comparative experimental approach:

1.  **Hybrid Architecture (HQCNN)**:
    *   **Feature Extractor**: EfficientNet-B0 (pre-trained on ImageNet) with frozen layers.
    *   **Quantum Classifier**: VQC implemented in **PennyLane**:
        *   **Qubits**: 4 (mapping typically to classes: Glioma, Meningioma, Pituitary, No Tumor).
        *   **Ansatz**: Strongly Entangling Layers ($L \in \{2, 4, 6\}$) to maximize expressibility.
        *   **Embedding**: Angle Embedding.
    *   **Integration**: Using `TorchLayer` to integrate the quantum circuit into the PyTorch workflow (Hybrid End-to-End).

2.  **Dataset**:
    *   **Source**: Public "Brain Tumor MRI Dataset".
    *   **Classes**: Glioma, Meningioma, Pituitary, No Tumor.
    *   **Preprocessing**: Resizing, normalization, and data augmentation.

3.  **Experimentation**:
    *   **Scenarios**: Training on 10%, 25%, 50%, and 100% of the dataset.
    *   **Metrics**: Accuracy, F1-Score, Training Time, Inference Latency.
    *   **Validation Strategy**: Stratified k-fold Cross-Validation ($k=5$).
    *   **Statistical Analysis**: ANOVA to determine significance of results.

## 💻 Technologies

*   **Language**: Python
*   **Deep Learning**: PyTorch
*   **Quantum Computing**: PennyLane
*   **Environment**: Google Colab / Local

## 👥 Team

*   **Student**: Frank Edward Daza Gonzalez
*   **Director**: Julian Hurtado López, PhD
*   **Co-director**: Alba Marcela Herrera Trujillo

---
*This repository contains the code and documentation for the Master's Thesis project.*
