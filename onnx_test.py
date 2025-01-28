import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from onnx2torch import convert
import onnx
dataset = '20Datasets/NMeta_DiAtom_5Feat/heladeeprt'

# Define Dataset
class MyDataset(Dataset):
    def __init__(self, sequences, retention):
        self.sequences = sequences
        self.retention = retention

    def __len__(self):
        return len(self.retention)

    def __getitem__(self, idx):
        return self.sequences[idx], self.retention[idx]

# Define Model
class MyNet(nn.Module):
    def __init__(self, input_size):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=49, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_size * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


# Initialize Data
train_x = np.load('../data_matrix/' + dataset + '/train_x.npy').astype(np.float32)
train_y = np.load('../data_matrix/' + dataset + '/train_y.npy').astype(np.float32)
test_x = np.load('../data_matrix/' + dataset + '/test_x.npy').astype(np.float32)
test_y = np.load('../data_matrix/' + dataset + '/test_y.npy').astype(np.float32)
dataset_train = MyDataset(train_x, train_y)
dataset_test = MyDataset(test_x, test_y)
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=500)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNet(input_size=62).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    for batch in dataloader_train:
        X, y = batch
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
pytorch_outputs = []
for batch in dataloader_test:
    X, y = batch
    X, y = X.to(device), y.to(device)
    output = model(X)
    pytorch_outputs.append(output.cpu().detach().numpy())
    loss = loss_function(output, y.view(-1, 1))
    print(f"Test Loss: {loss.item():.4f}")

# Save Model to ONNX
sample_input = torch.randn(341, 49, 62).to(device)
torch.onnx.export(model, sample_input, "model.onnx", opset_version=11)

# Load ONNX Model and Compare
test_input = test_x.astype(np.float32)  # Ensure correct dtype
session = ort.InferenceSession("model.onnx")
onnx_outputs = session.run(None, {session.get_inputs()[0].name: test_input})

# Compare Outputs
pytorch_outputs = np.vstack(pytorch_outputs)
difference = np.abs(pytorch_outputs - onnx_outputs[0])
average_difference = np.mean(difference)

results_df = pd.DataFrame({
    'PyTorch Outputs': pytorch_outputs.flatten(),
    'ONNX Outputs': onnx_outputs[0].flatten(),
    'Difference': difference.flatten()
})

print(results_df[:20])  # Print a preview of the first few rows
print("Average Difference:", average_difference)


onnx_model = onnx.load("model.onnx")
pytorch_onnx_model = convert(onnx_model)
pytorch_onnx_model.to(device)

optimizer = torch.optim.Adam(pytorch_onnx_model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in dataloader_train:
        X, y = batch
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = pytorch_onnx_model(X)
        loss = loss_function(output, y.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

sample_input = torch.randn(341, 49, 62).to(device)
torch.onnx.export(model, sample_input, "model.onnx", opset_version=11)
# Load ONNX Model and Compare
test_input = test_x.astype(np.float32)  # Ensure correct dtype
session = ort.InferenceSession("model.onnx")
onnx_outputs = session.run(None, {session.get_inputs()[0].name: test_input})

# Compare Outputs
pytorch_outputs = np.vstack(pytorch_outputs)
difference = np.abs(pytorch_outputs - onnx_outputs[0])
average_difference = np.mean(difference)

results_df = pd.DataFrame({
    'PyTorch Outputs': pytorch_outputs.flatten(),
    'ONNX Outputs': onnx_outputs[0].flatten(),
    'Difference': difference.flatten()
})

print(results_df[:20])  # Print a preview of the first few rows
print("Average Difference:", average_difference)