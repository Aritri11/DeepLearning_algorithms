#################################################Sigmoid function implementation from scratch######################################

##############################################Sigmoid function derivative implementation from scratch###############################

#Package import
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    s=1 / (1 + np.exp(-x))
    return s, np.mean(s)

# Derivative of sigmoid function
def sigmoid_derivative(x):
    sig,_ = sigmoid(x)
    return sig * (1 - sig)


def main():
    # Create an array of values from -10 to 10 for visualization
    x_values = np.linspace(-10, 10, 100)

    # Calculate sigmoid and its derivative for the given x values
    sigmoid_values,sigmoid_mean = sigmoid(x_values)
    print('Sigmoid:',sigmoid_values)
    sigmoid_derivative_values = sigmoid_derivative(x_values)
    print('Derivative of sigmoid: ',sigmoid_derivative_values)

    # Plotting the sigmoid function
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, sigmoid_values, label="Sigmoid Function", color='blue')
    plt.axhline(y=sigmoid_mean, color='green', linestyle='--', label=f"Mean = {sigmoid_mean:.2f}")
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.legend()
    plt.grid(True)

    # Plot of the derivative of the sigmoid function
    plt.subplot(1, 2, 2)
    plt.plot(x_values, sigmoid_derivative_values, label="Derivative of Sigmoid", color='red')
    plt.title("Derivative of Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid'(x)")
    plt.grid(True)

    # Show both plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# Sigmoid
# a) The minimum value is 0 and maximum value is 1
# b) No, the values are not zero centered, its centered around 0.5
# c) Smooth gradient for small inputs, flattens for large positive/negative values.
# - sigmoid is generally not used because of this, gradient becomes zero for very small or very large values,
# if gradient becomes zero then there is no update/learning thats happening then it becomes redundant