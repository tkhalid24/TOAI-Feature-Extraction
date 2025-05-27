import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report
import scipy.io

# ----------------------------
# Parameters
# ----------------------------
fs = 200  # Sampling frequency
window_sec = 1
window_size = fs * window_sec
stride = fs // 2  # 50% overlap

# ----------------------------
# Load and window the data
# ----------------------------
data = scipy.io.loadmat("bci.mat")
rf = data["rf"][:, :20]  # Right foot, use first 20 channels
rh = data["rh"][:, :20]  # Right hand, use first 20 channels

def create_windows(signal, label, window_size, stride):
    windows = []
    labels = []
    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end].T)  # Shape: (channels, time)
        labels.append(label)
    return windows, labels

X_rf, y_rf = create_windows(rf, 0, window_size, stride)
X_rh, y_rh = create_windows(rh, 1, window_size, stride)

X = np.array(X_rf + X_rh)  # shape: (n_samples, channels, time)
y = np.array(y_rf + y_rh)

# Normalize per channel
X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)

# Reshape for EEGNet: (samples, 1, channels, time)
X = X[:, np.newaxis, :, :]

# ----------------------------
# Train/Val/Test split
# ----------------------------
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)

# ----------------------------
# EEGNet Model Definition
# ----------------------------
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(20, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 1 * (window_size // 4), 2)  # Adjust for pooling output
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        return self.classify(x)

# ----------------------------
# Training Loop
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds, val_truths = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_truths.extend(yb.numpy())
    val_acc = accuracy_score(val_truths, val_preds)
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")

# ----------------------------
# Final Test Evaluation
# ----------------------------
model.eval()
test_preds, test_truths = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_truths.extend(yb.numpy())

test_acc = accuracy_score(test_truths, test_preds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(classification_report(test_truths, test_preds, target_names=["Right Foot", "Right Hand"]))
