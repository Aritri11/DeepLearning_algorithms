#####################################ReLU function implementation from scratch###############################################
#####################################ReLU function derivative implementation from scratch####################################

#Package import
import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    s = np.sum([np.exp(i) for i in z])
    exp_z = [float(np.exp(j) / s) for j in z]
    return exp_z

def softmax_derivative(z):
    s=softmax(z)
    jm=np.zeros((len(z),len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                jm[i][j]=s[i]*(1-s[i])
            elif i!=j:
                jm[i][j]=-(s[i]*s[j])
            else:
                print("ERROR!!")
    return jm


def main():
        # Create an array of values from -10 to 10 for visualization
        z_values = np.linspace(-10, 10, 100)
        sf=softmax(z_values)
        print("Softmax function values are: ",sf)
        sf_deri=softmax_derivative(z_values)
        print("Softmax derivatives are: ",sf_deri)

        sf_deri_diag=np.diagonal(sf_deri)
        print("Diagonal values of Softmax are: ",sf_deri_diag)
        softmax_mean = np.mean(sf)

        # Plot of the softmax function
        plt.figure(figsize=(12, 6))
        plt.plot(z_values, sf, label="Softmax Function", color='blue')
        plt.axhline(y=softmax_mean, color='green', linestyle='--', label=f"Mean = {softmax_mean:.2f}")
        plt.title("Softmax Function")
        plt.xlabel("x")
        plt.ylabel("softmax(x)")
        plt.legend()
        plt.grid(True)

        # # Show the plot
        plt.show()


if __name__ == "__main__":
    main()

# Softmax - computationally expensive
# a) The minimum value is 0 and maximum value is 1 (exponential growth for positive inputs)
# b) No, the values are not zero centered

# Relationship bw softmax and tanh
# tanh(z)=2*sigmoid(2z)-1

# zero centered = average of all function outputs will be zero