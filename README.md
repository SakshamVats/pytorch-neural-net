```
# Simple Binary Classifier with PyTorch

This project demonstrates a basic binary classification model built using **PyTorch**. It trains a small feedforward neural network on a synthetic dataset generated using `sklearn.datasets.make_classification`.

## 🧠 Model Architecture

- Input Layer: 10 features  
- Hidden Layer: 32 neurons with ReLU activation  
- Output Layer: 1 neuron with Sigmoid activation  
- Loss Function: Binary Cross Entropy Loss (BCELoss)  
- Optimizer: Adam

## 🛠 Features

- Data preprocessing using `StandardScaler`
- Train/test split
- Training loop with loss reporting
- Evaluation with accuracy on test data
- Optional GPU support

## 📦 Requirements

- Python 3.8+
- PyTorch
- scikit-learn

You can install the dependencies in a virtual environment with:

```bash
pip install torch torchvision torchaudio
pip install scikit-learn
```

## 🚀 Running the Model

```bash
python simple_neural_net.py
```

You’ll see output like:

```
Epoch 5/20, Loss: 0.5834  
Epoch 10/20, Loss: 0.4765  
Epoch 15/20, Loss: 0.3810  
Epoch 20/20, Loss: 0.3108  
Test Accuracy: 0.8550
```

## 📁 Project Structure

```
simple-nn/
├── simple_neural_net.py        # Training script
├── README.md      # Project readme      # Optional virtual environment
```

## 💡 Notes

- The model can be trained on GPU by using `model.to(device)` and moving the data to the same device.
- You can experiment with model depth, hidden neurons, or optimizer settings for better performance.
```

