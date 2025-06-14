import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Linear Model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        # Define a single linear layer: input feature is 1 (x), output feature is 1 (y)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # Forward pass applies the linear transformation
        return self.linear(x)

# 2. Prepare Training Data for y = 2x
# Generate some x values
X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
# Generate corresponding y values based on y = 2x
y_train = np.array([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]], dtype=np.float32)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)

# 3. Instantiate the Model, Loss Function, and Optimizer
model = SimpleLinearRegression()
# Mean Squared Error loss is suitable for regression
criterion = nn.MSELoss()
# Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Starting training of the linear model...")

# 4. Train the Model
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass: compute predicted y by passing X to the model
    outputs = model(X_train_tensor)
    # Compute loss
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad() # Clear gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update weights

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# 5. Verify the learned weights (optional)
# The ideal weight should be close to 2 for y=2x
# The ideal bias should be close to 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Learned parameter '{name}': {param.data.numpy()}")

# 6. Export the Model to ONNX format
onnx_filename = "linear.onnx"

# Create a dummy input tensor for ONNX export
# This input should have the same shape as the actual input your model expects
dummy_input = torch.randn(1, 1, requires_grad=True)

try:
    torch.onnx.export(model,                  # The PyTorch model
                      dummy_input,            # A dummy input to trace the model's computation graph
                      onnx_filename,          # Where to save the ONNX model (file path)
                      export_params=True,     # Store the trained parameter weights inside the model file
                      opset_version=11,       # The ONNX opset version to use
                      do_constant_folding=True, # Whether to execute constant folding for optimization
                      input_names=['input'],  # Names for the model's input
                      output_names=['output'],# Names for the model's output
                      dynamic_axes={'input' : {0 : 'batch_size'},    # Specify variable batch size
                                    'output' : {0 : 'batch_size'}})
    print(f"Model successfully exported to {onnx_filename}")

except Exception as e:
    print(f"Error exporting model to ONNX: {e}")

print(f"You can now use '{onnx_filename}' with ONNX Runtime or other ONNX-compatible tools.")
