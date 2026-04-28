# Neural Network Models for Time-Series Sensor Classification

> Benchmarking CNN, RNN (LSTM), and Autoencoder-based classifiers on multivariate time-series sensor data — with systematic hyperparameter tuning across all three architectures.

---

## Overview

This project classifies multivariate time-series sensor data into 11 classes using three different deep learning approaches: a 1D CNN, an LSTM-based RNN, and a convolutional Autoencoder used as a feature extractor. Each model is developed and tuned independently, and their best configurations are compared on a held-out test set.

---

## Dataset

- 8 sensors, 55 timesteps per sample, 11 classes (0–10)
- Input shape: `[batch, 55, 8]`
- EDA showed correlated sensor behavior across timesteps and low class separability from mean time-series alone — motivating deep non-linear models over simple baselines

---

## Files

| File | Description |
|---|---|
| `Time_Series_Sensor_Dataset.ipynb` | Data loading, EDA, and visualization |
| `CNN_Final.ipynb` | 1D CNN — architecture, hyperparameter search, training |
| `RNN.ipynb` | LSTM-based RNN — architecture, hyperparameter search, training |
| `Autoencoders_Final.ipynb` | Convolutional Autoencoder pre-training + classifier fine-tuning |
| `Neural_Network_Models_for_Time_Series_Classification.ipynb` | Combined comparison notebook |
| `Deep_Learning_Project_-_Time-Series_Sensor_Data.pdf` | Project report with architectures, tuning rationale, and results |

---

## Models & Results

| Model | Test Accuracy | Best Config |
|---|---|---|
| **Autoencoder** | **94.84%** | Adam, lr=0.001, pre-activation BatchNorm, LRScheduler, Dropout=0.3 |
| CNN | 93.93% | Adam, lr=0.001, pre-activation BatchNorm, batch=32 |
| RNN (LSTM) | 86.24% | Adam, lr=0.0001, hidden=256, batch=32 |

---

## Architectures

### CNN
Two `Conv1d` blocks (8→32→64 channels) with BatchNorm, ReLU, and MaxPool, followed by `AdaptiveAvgPool1d` and a two-layer classifier head. Input is permuted from `[B, 55, 8]` to `[B, 8, 55]` for 1D convolution.

### RNN
Single-layer LSTM (hidden=256) operating on the full sequence; final hidden state passed to a linear classifier. Best performance without adding ReLU to the FC layer.

### Autoencoder
Two-phase training:
1. **Pre-train** encoder-decoder (Conv1d 8→16→32, latent dim=10, ConvTranspose1d decoder) with MSE reconstruction loss
2. **Freeze encoder**, attach a classifier head (Linear 10→32→11 with BatchNorm and Dropout), train classifier with cross-entropy

The pre-trained encoder gives a strong initialization, which is why the autoencoder reaches the highest accuracy with minimal tuning.

---

## Hyperparameter Tuning Summary

**CNN** — SGD was noisy at lr=0.001 and underfit at lr=0.0001 with momentum. Switching to Adam improved convergence significantly. Pre-activation BatchNorm after each conv layer provided the final accuracy gain.

**RNN** — Adam with lr=0.0001 throughout. Progressively increased hidden size (64→128→256) until accuracy plateaued. Adding ReLU to the FC layer slightly hurt performance so it was removed.

**Autoencoder** — Started strong due to pre-trained encoder. Added `ReduceLROnPlateau` scheduler for stability, then pre-activation BatchNorm for regularization. Light Dropout (0.3) corrected mild overfitting observed in train/validation curves.

---

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib
```

All notebooks are self-contained and run sequentially. GPU is used automatically if available.
