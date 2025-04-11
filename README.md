```
# 🧠 Simple Binary Classifier with PyTorch

This project demonstrates a basic **binary classification** model using **PyTorch**. It uses a synthetic dataset generated with `sklearn.datasets.make_classification`.

## 📌 Model Summary

- **Input Layer**: 10 features  
- **Hidden Layer**: 32 neurons with ReLU  
- **Output Layer**: 1 neuron with Sigmoid  
- **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`)  
- **Optimizer**: Adam

## ⚙️ Features

- Preprocessing with `StandardScaler`
- Train/test splitting
- Custom training loop with loss tracking
- Accuracy evaluation
- Compatible with **CPU** and **GPU**

## 📦 Requirements

- Python 3.8+
- PyTorch
- scikit-learn

### 💾 Installation

Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On macOS/Linux

pip install torch torchvision torchaudio
pip install scikit-learn
```

## 🚀 How to Run

```bash
python simple_neural_net.py
```

### ✅ Example Output

```
Epoch 5/20, Loss: 0.5834  
Epoch 10/20, Loss: 0.4765  
Epoch 15/20, Loss: 0.3810  
Epoch 20/20, Loss: 0.3108  
Test Accuracy: 0.8550
```

## 📁 Project Structure

```
simple-binary-classifier/
├── simple_neural_net.py        # Main training script
├── README.md      # This file         # Optional virtual environment
```

## 💡 Notes

- Move your model and data to GPU using `.to("cuda")` if available.
- You can tweak the model size, optimizer, and training loop to explore improvements.
```
