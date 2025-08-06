Deep Learning Activation Functions

This repository contains Python implementations of the most commonly used activation functions in deep learning:

1. Sigmoid: f(x) = 1 / (1 + exp(-x))

2. ReLU (Rectified Linear Unit): f(x) = max(0, x)

3. Tanh (Hyperbolic Tangent): f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

4. Leaky ReLU: f(x) = x if x > 0 else alpha * x (with default alpha=0.01)

5. Softmax: f(xi) = exp(xi) / sum(exp(xj)) for all elements in x
