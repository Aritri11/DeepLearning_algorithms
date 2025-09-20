

import numpy as np
#ReLU Function
def ReLU(x):
    r=np.maximum(0,x)
    return r,np.mean(r)

def softmax(z):
    s = np.sum([np.exp(i) for i in z])
    exp_z = [float(np.exp(j) / s) for j in z]
    return exp_z

def ReLU_grad(z):
    return np.where(z > 0, 1, 0)  # Derivative of ReLU

def MSE_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSE_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)  # Gradient of MSE

# Backward pass using computational graph concepts
def backward_pass(y_true, weights, biases, activations, zs):
    dw = [0] * len(weights)  # Gradients of weights
    db = [0] * len(biases)   # Gradients of biases

    # Global gradient from loss w.r.t output (start of backprop)
    delta = MSE_grad(y_true, activations[-1])  # dL/da

    for l in reversed(range(len(weights))):
        print(f"\nBackward pass at Layer {l+1}")

        # Local gradient: activation derivative
        if l == len(weights) - 1:
            # Output layer: could be ReLU or Softmax
            # Assume ReLU here for simplicity (or modify for Softmax)
            local_grad = ReLU_grad(zs[l])
        else:
            local_grad = ReLU_grad(zs[l])

        # Chain Rule: Global grad = local grad * prev global grad
        delta = delta * local_grad  # Element-wise

        # Gradients for weights and biases
        dw[l] = np.outer(delta, activations[l])  # dL/dW
        db[l] = delta  # dL/db

        # Update delta to propagate to previous layer
        delta = np.dot(weights[l].T, delta)  # dL/da of previous layer

    return dw, db

def forward_pass(x, w, b):
    a = x
    activations = [a]  # Store input layer activations
    zs = []            # Store z-values per layer

    for j in range(len(w)):
        z = np.dot(w[j], a) + b[j]
        zs.append(z)

        if j < (len(w) - 1):
            a = ReLU(z)
        else:
            prompt = input("Enter R/S for final activation: ")
            if prompt == "S":
                a = softmax(z)
            elif prompt == "R":
                a = ReLU(z)
            else:
                print("Invalid, defaulting to ReLU")
                a = ReLU(z)

        activations.append(a)

    return a, activations, zs

def main():
    q = int(input("Enter the number of inputs: "))
    x = np.array([float(input(f"Enter value for input {i+1}: ")) for i in range(q)])

    l = int(input("Enter the number of layers: "))
    layer_sizes = [q]
    for i in range(l):
        layer_sizes.append(int(input(f"Enter neurons in layer {i+1}: ")))

    w = [np.random.rand(layer_sizes[i+1], layer_sizes[i]) for i in range(l)]
    b = [np.random.rand(layer_sizes[i+1]) for i in range(l)]

    output, activations, zs = forward_pass(x, w, b)
    print("Network output:", output)

    y_true = np.array([float(input(f"Enter expected output value {i+1}: ")) for i in range(len(output))])

    dw, db = backward_pass(y_true, w, b, activations, zs)
    print("\nWeight Gradients (dW):")
    for g in dw:
        print(g)
    print("\nBias Gradients (dB):")
    for g in db:
        print(g)

if __name__ == "__main__":
    main()
