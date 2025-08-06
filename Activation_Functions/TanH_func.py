#####################################TanH function implementation from scratch###############################################
#####################################TanH function derivative implementation from scratch####################################

#Package import
import numpy as np
import matplotlib.pyplot as plt


#Tanh Function
def tanh(x):
    tan=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return tan, np.mean(tan)

#Derivative of Tanh function
def tanh_derivative(x):
    t,_=tanh(x)
    return 1-(t**2)


def main():
    # Create an array of values from -10 to 10 for visualization
    x_values = np.linspace(-10, 10, 100)

    # Calculate TanH and its derivative for the given x values
    tanh_values,tanh_mean = tanh(x_values)
    print('TanH:', tanh_values)
    tanh_derivative_values = tanh_derivative(x_values)
    print('Derivative of tanh: ', tanh_derivative_values)

    # Plot of the TanH function
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, tanh_values, label="TanH Function", color='blue')
    plt.axhline(y=tanh_mean, color='green', linestyle='--', label=f"Mean = {tanh_mean:.2f}")
    plt.title("TanH Function")
    plt.xlabel("x")
    plt.ylabel("TanH(x)")
    plt.legend()
    plt.grid(True)

    # Plot of the derivative of the TanH function
    plt.subplot(1, 2, 2)
    plt.plot(x_values, tanh_derivative_values, label="Derivative of Sigmoid", color='red')
    plt.title("Derivative of TanH Function")
    plt.xlabel("x")
    plt.ylabel("TanH'(x)")
    plt.grid(True)

    # Show both plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Tanh
# a) The minimum value is -1 and maximum value is 1
# b) Yes, the values are zero centered
# c) Smooth gradient for small inputs, flattens for large positive/negative values.