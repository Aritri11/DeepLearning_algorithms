#####################################ReLU function implementation from scratch###############################################
#####################################ReLU function derivative implementation from scratch####################################

#Package import
import numpy as np
import matplotlib.pyplot as plt


#ReLU Function
def ReLU(x):
    r=np.maximum(0,x)
    return r,np.mean(r)

#Derivative of ReLU function
def ReLU_derivative(x):
    return np.where(x > 0, 1.0, 0.0)



def main():
    # Create an array of values from -10 to 10 for visualization
    x_values = np.linspace(-10, 10, 100)

    # Calculate ReLU and its derivative for the given x values
    relu_values, relu_mean = ReLU(x_values)
    print('relu:', relu_values)
    relu_derivative_values = ReLU_derivative(x_values)
    print('Derivative of Relu: ', relu_derivative_values)

    # Plot of the ReLU function

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, relu_values, label="Relu Function", color='blue')
    plt.axhline(y=relu_mean, color='green', linestyle='--', label=f"Mean = {relu_mean:.2f}")
    plt.title("RELU Function")
    plt.xlabel("x")
    plt.ylabel("RELU(x)")
    plt.legend()
    plt.grid(True)

    # Plot of the derivative of the RELU function
    plt.subplot(1, 2, 2)
    plt.plot(x_values, relu_derivative_values, label="Derivative of RELU", color='red')
    plt.title("Derivative of RELU Function")
    plt.xlabel("x")
    plt.ylabel("RELU'(x)")
    plt.grid(True)

    # Show both plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# ReLU - normally used
# a) The minimum value is 0 and maximum value is inf (Linear growth for positive inputs)
# b) No, the values are not zero centered, the values are always more than 0
# c) Gradient:  For positive values: gradient = 1
#    Gradient:  For negative values: gradient = 0
# cost of computing maximum is less than computing exponential
# - so this is better and computationally less expensive than other activation func
# Always keep in mind
# - time
# - space
# - computational power