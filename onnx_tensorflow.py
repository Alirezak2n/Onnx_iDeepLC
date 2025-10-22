import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from onnx2torch import convert
import onnx
import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # Suppress TracerWarnings

dataset = '20Datasets/NMeta_DiAtom_5Feat/heladeeprt'

# Define Dataset
class MyDataset(Dataset):
    def __init__(self, sequences, retention):
        self.sequences = sequences
        self.retention = retention

    def __len__(self):
        return len(self.retention)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.retention[idx], dtype=torch.float32)

# Initialize Data
train_x = np.load('../data_matrix/' + dataset + '/train_x.npy').astype(np.float32)
train_y = np.load('../data_matrix/' + dataset + '/train_y.npy').astype(np.float32)
test_x = np.load('../data_matrix/' + dataset + '/test_x.npy').astype(np.float32)
test_y = np.load('../data_matrix/' + dataset + '/test_y.npy').astype(np.float32)
dataset_train = MyDataset(train_x, train_y)
dataset_test = MyDataset(test_x, test_y)
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=500)

# Define TensorFlow Model
inputs = tf.keras.Input(shape=(49, 62))
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation=None)(x)
outputs = tf.keras.layers.Reshape((1, 1))(outputs)  # Ensures proper shape to prevent squeeze

tf_model = tf.keras.Model(inputs, outputs)
tf_model.compile(optimizer='adam', loss='mse')

# Train TensorFlow Model for 10 epochs
print("Training TensorFlow model...")
tf_model.fit(train_x, train_y, epochs=10, batch_size=128, verbose=1)

# Convert TensorFlow Model to ONNX with input signature
onnx_model_path = "tf_model.onnx"
spec = (tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype, name=tf_model.inputs[0].name),)
tf2onnx.convert.from_keras(tf_model, input_signature=spec, output_path=onnx_model_path)

# Load ONNX Model and Compare Before Transfer Learning
test_input = test_x.astype(np.float32)
session = ort.InferenceSession(onnx_model_path)
onnx_outputs_before = session.run(None, {session.get_inputs()[0].name: test_input})

# Convert ONNX Model to PyTorch
onnx_model = onnx.load(onnx_model_path)
pytorch_onnx_model = convert(onnx_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_onnx_model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(pytorch_onnx_model.parameters(), lr=0.001)

# Train for 100 epochs (Transfer Learning)
print("Training PyTorch model after ONNX conversion...")
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

# Save Fine-Tuned Model to ONNX
torch.onnx.export(pytorch_onnx_model, torch.randn(1, 49, 62).to(device), "fine_tuned_model.onnx", opset_version=11,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# Load ONNX Model After Transfer Learning
session = ort.InferenceSession("fine_tuned_model.onnx")
onnx_outputs_after = session.run(None, {session.get_inputs()[0].name: test_input})

# Compare Outputs Before and After Transfer Learning
difference_before = np.abs(onnx_outputs_before[0] - test_y)
difference_after = np.abs(onnx_outputs_after[0] - test_y)

results_df_before = pd.DataFrame({
    'ONNX Outputs Before': onnx_outputs_before[0].flatten(),
    'True Labels': test_y.flatten(),
    'Difference Before': difference_before.flatten()
})

results_df_after = pd.DataFrame({
    'ONNX Outputs After': onnx_outputs_after[0].flatten(),
    'True Labels': test_y.flatten(),
    'Difference After': difference_after.flatten()
})

print("Performance Before Transfer Learning:")
print(results_df_before[:20])
print("Average Difference (Before Transfer Learning):", np.mean(difference_before))

print("Performance After Transfer Learning:")
print(results_df_after[:20])
print("Average Difference (After Transfer Learning):", np.mean(difference_after))
