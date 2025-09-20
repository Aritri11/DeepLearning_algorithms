import numpy as np

# -------------------------
# Utilities
# -------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def mse_loss(y_pred, y_true):
    loss = np.mean((y_pred - y_true) ** 2)
    dloss = 2 * (y_pred - y_true) / y_true.size
    return loss, dloss

# -------------------------
# Batch Normalization (FC / vector)
# -------------------------
class BatchNorm:
    """
    Batch Normalization for 2D inputs X: shape (N, D)
    Maintains running mean/var for inference and supports forward/backward.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        # learnable params (scale and shift)
        self.gamma = np.ones((dim,), dtype=float)
        self.beta = np.zeros((dim,), dtype=float)

        # running stats (for inference)
        self.running_mean = np.zeros((dim,), dtype=float)
        self.running_var = np.ones((dim,), dtype=float)

        # cache used by backward
        self.cache = None

    def forward(self, x, training=True):
        """
        x: (N, D)
        returns: out (N, D)
        """
        if training:
            mu = np.mean(x, axis=0)             # (D,)
            var = np.var(x, axis=0)            # (D,)
            std = np.sqrt(var + self.eps)      # (D,)
            x_hat = (x - mu) / std             # (N, D)
            out = self.gamma * x_hat + self.beta

            # update running stats (exponential moving average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var

            # save for backward
            self.cache = (x, x_hat, mu, var, std)
            return out
        else:
            # inference: use running stats
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
            return out

    def backward(self, dout):
        """
        dout: upstream gradient (N, D)
        returns: dx (N, D), dgamma (D,), dbeta (D,)
        """
        if self.cache is None:
            raise RuntimeError("No cache found. Did you forget to call forward(training=True)?")

        x, x_hat, mu, var, std = self.cache
        N, D = x.shape

        # gradients for gamma and beta
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        # gradient w.r.t. normalized input
        dx_hat = dout * self.gamma   # (N, D)

        # compact formula for gradient w.r.t. x
        sum_dxhat = np.sum(dx_hat, axis=0)                         # (D,)
        sum_dxhat_xhat = np.sum(dx_hat * x_hat, axis=0)            # (D,)

        dx = (1.0 / N) * (1.0 / std) * (N * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat)

        return dx, dgamma, dbeta

# -------------------------
# Layer Normalization
# -------------------------
class LayerNorm:
    """
    Layer Normalization for 2D inputs X: shape (N, D)
    Norm is computed per-sample across features (axis=1).
    Gamma and beta are shape (D,) and shared across batch.
    """
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps

        # learnable params
        self.gamma = np.ones((dim,), dtype=float)
        self.beta = np.zeros((dim,), dtype=float)

        self.cache = None

    def forward(self, x):
        """
        x: (N, D)
        return: out (N, D)
        """
        mu = np.mean(x, axis=1, keepdims=True)          # (N,1)
        var = np.var(x, axis=1, keepdims=True)          # (N,1)
        std = np.sqrt(var + self.eps)                   # (N,1)
        x_hat = (x - mu) / std                          # (N,D)
        out = self.gamma * x_hat + self.beta            # broadcast gamma/beta over batch

        self.cache = (x, x_hat, mu, var, std)
        return out

    def backward(self, dout):
        """
        dout: (N, D)
        returns: dx (N, D), dgamma (D,), dbeta (D,)
        """
        if self.cache is None:
            raise RuntimeError("No cache found. Did you forget to call forward()?")

        x, x_hat, mu, var, std = self.cache
        N, D = x.shape

        # grads for gamma and beta (sum over batch)
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        # per-sample normalization along features
        dx_hat = dout * self.gamma  # (N,D)
        sum_dxhat = np.sum(dx_hat, axis=1, keepdims=True)               # (N,1)
        sum_dxhat_xhat = np.sum(dx_hat * x_hat, axis=1, keepdims=True)  # (N,1)

        dx = (1.0 / D) * (1.0 / std) * (D * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat)
        return dx, dgamma, dbeta

# -------------------------
# Demo / Example usage
# -------------------------
def demo_normalizers():
    # Toy dataset: 4 samples, 3 features
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ], dtype=float)

    # Fake gradient coming from next layer (same shape as X)
    d_out = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.3, 0.4, 0.1],
        [0.5, 0.2, 0.2]
    ], dtype=float)

    lr = 0.01  # learning rate for simple parameter update demo

    print("=== BatchNorm demo ===")
    bn = BatchNorm(dim=3)
    out_bn = bn.forward(X, training=True)
    print("Forward output (BatchNorm):\n", out_bn)

    dx_bn, dgamma_bn, dbeta_bn = bn.backward(d_out)
    print("\nGradients from BatchNorm backward:")
    print("dx (shape {}) =\n{}".format(dx_bn.shape, dx_bn))
    print("dgamma =", dgamma_bn)
    print("dbeta  =", dbeta_bn)

    # simple SGD update on gamma/beta to show parameter change (not a full optimizer)
    bn.gamma -= lr * dgamma_bn
    bn.beta  -= lr * dbeta_bn
    print("\nUpdated gamma (after SGD step):", bn.gamma)
    print("Updated beta  (after SGD step):", bn.beta)

    print("\n=== LayerNorm demo ===")
    ln = LayerNorm(dim=3)
    out_ln = ln.forward(X)
    print("Forward output (LayerNorm):\n", out_ln)

    dx_ln, dgamma_ln, dbeta_ln = ln.backward(d_out)
    print("\nGradients from LayerNorm backward:")
    print("dx (shape {}) =\n{}".format(dx_ln.shape, dx_ln))
    print("dgamma =", dgamma_ln)
    print("dbeta  =", dbeta_ln)

    # simple SGD update on gamma/beta for LayerNorm
    ln.gamma -= lr * dgamma_ln
    ln.beta  -= lr * dbeta_ln
    print("\nUpdated gamma (after SGD step):", ln.gamma)
    print("Updated beta  (after SGD step):", ln.beta)

if __name__ == "__main__":
    demo_normalizers()

