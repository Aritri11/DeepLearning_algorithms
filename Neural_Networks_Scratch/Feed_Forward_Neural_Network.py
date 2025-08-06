####################################### Feed Forward Neural Network (FNN):- Scratch Implementation #####################
################################### Dual implementation of user-interactive & pre-defined inputs #################################

#Package import
import numpy as np
import random

# ReLU Activation Function
def ReLU(z):
    return np.maximum(0, z)  # Returns max(0, z) for each element

# Softmax Activation Function
def softmax(z):
    s = np.sum([np.exp(i) for i in z])  # Sum of exponentials for normalization
    exp_z = [float(np.exp(j) / s) for j in z]  # Normalized exponentials (probabilities)
    return exp_z

# Forward propagation through the neural network
def forward_pass(x, w, b):
    a = x  # Initialize activations with input vector

    for j in range(len(w)):
        z = np.dot(w[j], a) + b[j]  # Compute weighted sum + bias for layer j

        # Print layer-wise details for debugging
        print(f"\nLayer {j + 1}")
        print("Weights:")
        print(w[j])
        print("Biases:")
        print(b[j])
        print("Z (Weighted sum):")
        print(z)

        # If not the final layer, apply ReLU
        if j < (len(w) - 1):
            a = ReLU(z)
        else:
            # For the final layer, allow user to choose between ReLU or Softmax
            print(f"1. Type R - if you want to use ReLU activation function at output layer\n2. Type S - to use Softmax activation function at the output layer")
            prompt = input("Enter R/S: ")
            if prompt == "S":
                a = softmax(z)
            elif prompt == "R":
                a = ReLU(z)
            else:
                print("Enter correct prompt")
                forward_pass(x, w, b)  # Recursively call function if invalid input
    return a  # Final output

# Function to define the number of neurons in each layer
def layers(l, input_size):
    neu = [int(input(f"Enter the number of neurons in layer {k + 1}: ")) for k in range(l)]
    neu.insert(0, input_size)  # Add input size as first layer
    return neu

# Function to generate or accept user-defined weights and biases
def weights(n, l):
    w = []
    b = []
    choice = input("\nDo you want to manually enter weights and biases? (y/n): ").strip().lower()

    for i in range(l):
        if choice == "y":
            # User-defined weights
            print(f"\nEnter weights for Layer {i + 1} ({n[i + 1]}x{n[i]}):")
            weight = []
            for j in range(n[i + 1]):
                row = []
                for k in range(n[i]):
                    val = float(input(f"Weight[{j}][{k}]: "))
                    row.append(val)
                weight.append(row)
            weight = np.array(weight)

            # User-defined biases
            print(f"\nEnter biases for Layer {i + 1} (size {n[i + 1]}):")
            bias = []
            for j in range(n[i + 1]):
                val = float(input(f"Bias[{j}]: "))
                bias.append(val)
            bias = np.array(bias)
        else:
            # Random weights and biases between 0.001 and 1
            weight = np.array([random.uniform(0.001, 1) for _ in range(n[i] * n[i + 1])])
            weight = weight.reshape(n[i + 1], n[i])
            bias = np.array([random.uniform(0.001, 1) for _ in range(n[i + 1])])

        w.append(weight)
        b.append(bias)
    return w, b

# Main driver function
def main():
    # Get user input for network input vector
    q = int(input("Enter the number of inputs you want: "))
    x = [float(input(f"Enter the values of each input {i + 1}: ")) for i in range(q)]

    # Get number of layers
    l = int(input("Enter the number of layers you want: "))

    x = np.array(x)  # Convert input to NumPy array
    n = layers(l, len(x))  # Get architecture of network
    w, b = weights(n, l)  # Initialize weights and biases
    output = forward_pass(x, w, b)  # Perform forward pass
    print("Output is: ", output)

# Entry point of the program
if __name__ == "__main__":
    main()



#NOTE:
#The same code works for XOR function when inputs of all weights,biases as well as x values are given according to the XOR truth table
