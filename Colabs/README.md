# 🧠 Hybrid CNN-VQC for Brain Tumor Diagnosis - Colab Implementation Guide

This guide provides a step-by-step walkthrough to implement the research project: **"Evaluation of Data Efficiency in a Hybrid CNN-VQC Architecture for Brain Tumor Diagnosis in Magnetic Resonance Imaging"**.

The implementation is designed to be executed in **Google Colab**, taking advantage of free GPU resources and easy integration with Quantum libraries like PennyLane.

---

## 🚀 1. Environment Setup

Before starting the coding, you need to set up the Colab environment.

### **Step 1.1: Enable GPU**
1. Go to `Runtime` > `Change runtime type`.
2. Select **T4 GPU** (or better if available via Colab Pro).
3. Click `Save`.

### **Step 1.2: Install Dependencies**
You will need to install `PennyLane` for quantum computing simulations, along with standard deep learning libraries.

```python
!pip install pennylane --quiet
!pip install torch torchvision torchaudio --quiet
!pip install matplotlib seaborn scikit-learn pandas numpy --quiet
```

---

## 📂 2. Data Preparation (Phase 2)

We need to load and preprocess the **Brain Tumor MRI Dataset**.

### **Option A: Using Google Drive (Recommended for Colab)**
1.  Upload your dataset as a **zip file** (e.g., `dataset.zip`) to your Google Drive.
2.  In the notebook, you will mount your Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  Set the path to your zip file in the notebook:
    ```python
    DRIVE_ZIP_PATH = '/content/drive/MyDrive/dataset.zip'
    ```
4.  The notebook will automatically copy the zip to the Colab runtime and unzip it. This is much faster than reading files directly from Drive.

### **Option B: Using Local Data**
If you are running locally or have already uploaded the folder structure:
You should have the dataset located in `Colabs/data/`.
The expected structure is:
```
Colabs/
  data/
    Training/
      glioma/
      meningioma/
      ...
    Testing/
      ...
```

### **Step 2.2: Data Preprocessing & Loading**
We must resize images to **224x224**, normalize them using ImageNet statistics, and prepare DataLoaders.

*   **Classes**: glioma, meningioma, pituitary, notumor.
*   **Transformation**:
    *   Resize: `(224, 224)`
    *   ToTensor
    *   Normalize: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The notebook will set data_dir based on Option A or B
# data_dir = '../data/Training' 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
```

---

## 🏗️ 3. Architecture Construction (Phase 1)

This phase involves building two models: the Classical Baseline (EfficientNet) and the Hybrid CNN-VQC.

### **Step 3.1: Classical Baseline (EfficientNet-B0)**
We use a pre-trained EfficientNet-B0 and replace the classifier head.

```python
import torchvision.models as models
import torch.nn as nn

def get_classical_model(num_classes=4):
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
```

### **Step 3.2: Hybrid CNN-VQC Architecture**
We replace the final classical layer with a **Variational Quantum Circuit (VQC)**.

#### **A. Define the Quantum Layer (PennyLane)**
*   **Qubits**: 4 (matching the 4 classes).
*   **Embedding**: Angle Embedding.
*   **Ansatz**: Strongly Entangling Layers.

```python
import pennylane as qml

n_qubits = 4
n_layers = 2  # Adjustable (Try 2, 4, 6)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

#### **B. Integrate with PyTorch**
Use `qml.qnn.TorchLayer` to convert the QNode into a Torch layer.

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Freeze features
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # Reduce dimensionality to match n_qubits
        self.pre_net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, n_qubits) # EfficientNet-B0 output is 1280
        )
        
        # Quantum Layer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_net(x)
        x = self.q_layer(x) # Output is -1 to 1 per class
        # Ideally, add softmax or mapping here if needed, 
        # or use CrossEntropyLoss which handles logits (though Q-output is restricted)
        return x 
```

---

## 🔬 4. Experimentation Pipeline (Phase 3)

The core of the research is evaluating performance under data scarcity.

### **Step 4.1: Data Scarcity Simulation**
Create a function to subset the dataset.

```python
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def get_subset(dataset, percentage):
    targets = [s[1] for s in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=percentage, random_state=42)
    train_idx, _ = next(sss.split(np.zeros(len(targets)), targets))
    return Subset(dataset, train_idx)

# Example: Get 10% of data
subset_10 = get_subset(dataset, 0.1)
```

### **Step 4.2: Stratified K-Fold Cross-Validation**
Implement a loop to train and validate 5 times for each scenario.

```python
from sklearn.model_selection import StratifiedKFold

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Pseudo-code for the loop
for fold, (train_ids, val_ids) in enumerate(skf.split(X, y)):
    # 1. Create DataLoaders for this fold
    # 2. Initialize Model (Fresh for each fold)
    # 3. Train for N epochs
    # 4. Log Metrics (Accuracy, F1, Loss)
```

### **Step 4.3: Training Loop**
Standard PyTorch training loop.

*   **Loss Function**: `nn.CrossEntropyLoss()`
*   **Optimizer**: `optim.Adam(model.parameters(), lr=0.001)`

---

## 📊 5. Evaluation & Analysis

### **Step 5.1: Metrics**
Track the following for every experiment:
1.  **Training Time** (per epoch and total).
2.  **Accuracy** (Train vs Validation).
3.  **F1-Score** (Weighted).
4.  **Inference Latency** (Time to predict a single batch).

### **Step 5.2: Visualization**
Use `matplotlib` to plot:
*   Learning Curves (Loss/Accuracy vs Epochs).
*   Confusion Matrices.
*   Comparison Bar Charts (Hybrid vs Classical at 10%, 25%, 50% data).

---

## 📚 Recommended Resources

*   **PennyLane QML Tutorials**: [https://pennylane.ai/qml/demonstrations](https://pennylane.ai/qml/demonstrations)
*   **PyTorch Transfer Learning**: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
*   **EfficientNet Paper**: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
