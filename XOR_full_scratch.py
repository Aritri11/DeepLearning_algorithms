import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


np.random.seed(42)
input_size = X.shape[1]
output_size = 1
W = np.random.randn(input_size, output_size)
b = np.zeros((1, output_size))


lr = 0.1
epochs = 1000
losses = []

# Training loop
for epoch in range(epochs):
    # ===== Forward pass =====
    z = np.dot(X, W) + b
    a = sigmoid(z)

    # Compute loss
    loss = mse_loss(y, a)
    losses.append(loss)

    # ===== Backward pass =====
    # Global gradient from loss
    dL_da = mse_grad(y, a)
    # Local gradient from activation
    da_dz = sigmoid_grad(z)
    # Chain rule: global gradient for z
    dL_dz = dL_da * da_dz

    # Gradients for weights and bias
    dW = np.dot(X.T, dL_dz)
    db = np.sum(dL_dz, axis=0, keepdims=True)

    # ===== Update parameters =====
    W -= lr * dW
    b -= lr * db
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

print("\nFinal predictions after training:")
print(a)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.show()
