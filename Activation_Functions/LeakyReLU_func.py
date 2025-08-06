#####################################LeakyReLU function implementation from scratch###############################################
#####################################LeakyReLU function derivative implementation from scratch####################################

#Package import
import numpy as np
import matplotlib.pyplot as plt


#LeakyReLU Function
def leaky_relu(x):
    alpha = 0.1
    return np.maximum(x, alpha * x)

#Derivative of LeakyReLU function
def leaky_relu_derivative(x):
    alpha = 0.1
    return np.where(x > 0, 1.0, alpha)



def main():
    # Create an array of values from -10 to 10 for visualization
    x_values = np.linspace(-10, 10, 100)

    # Calculate LeakyReLU and its derivative for the given x values
    leaky_relu_values = leaky_relu(x_values)
    print('Leaky relu:', leaky_relu_values)
    leaky_relu_derivative_values = leaky_relu_derivative(x_values)
    print('Derivative of leaky Relu: ', leaky_relu_derivative_values)
    leaky_relu_mean = np.mean(leaky_relu_values)

    # Plot of the LeakyReLU function
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, leaky_relu_values, label="Leaky Relu Function", color='blue')
    plt.axhline(y=leaky_relu_mean, color='green', linestyle='--', label=f"Mean = {leaky_relu_mean:.2f}")
    plt.title("Leaky RELU Function")
    plt.xlabel("x")
    plt.ylabel("Leaky RELU(x)")
    plt.legend()
    plt.grid(True)

    # Plot of the derivative of the Leaky RELU function
    plt.subplot(1, 2, 2)
    plt.plot(x_values, leaky_relu_derivative_values, label="Derivative of Leaky RELU", color='red')
    plt.title("Derivative of Leaky RELU Function")
    plt.xlabel("x")
    plt.ylabel("Leaky RELU'(x)")
    plt.grid(True)

    # Show both plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Leaky ReLU
# a) The minimum value is -0.01 and maximum value is inf (Linear growth for positive inputs)
# b) No, the values are not zero centered
# c) Gradient:  For positive values: gradient = 1
#    Gradient:  For negative values: gradient = 0.01
