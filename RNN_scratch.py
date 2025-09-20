import numpy as np

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.Wx = np.random.randn(hidden_dim, input_dim) * 0.01   # Weight matrix for input x
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.01   # Weight matrix for hidden state h
        self.b = np.zeros((hidden_dim, 1)) #initialization of bias
        self.Wyh = np.random.randn(output_dim, hidden_dim) * 0.01  # Weight matrix to compute output y from hidden state h

    #Forward pass
    def forward(self, inputs):
        time_steps = inputs.shape[0]
        h = np.zeros((time_steps + 1, self.hidden_dim))
        y = np.zeros((time_steps, self.output_dim))

        #Loop to compute the hidden state at time t  time step
        for t in range(time_steps):
            x_t = inputs[t].reshape(self.input_dim, 1)
            h[t + 1] = np.tanh(self.Wx @ x_t + self.Wh @ h[t].reshape(self.hidden_dim, 1) + self.b).reshape(
                self.hidden_dim)
            y[t] = (self.Wyh @ h[t + 1].reshape(self.hidden_dim, 1)).reshape(self.output_dim)

        return h[1:], y

#To take the inputs from user
def user_input():
    time_steps = int(input("Enter the number of time steps: "))
    input_dim = int(input("Enter the dimension of the input(x) vector at each time step: "))
    inputs = []
    print(f"Enter the input(x) vector values for each of the {time_steps} time steps:")
    for t in range(time_steps):
        vec_str = input(f"Time step {t + 1} (space-separated {input_dim} values): ")
        vec = list(map(float, vec_str.strip().split()))
        if len(vec) != input_dim:
            raise ValueError(f"Expected {input_dim} values but got {len(vec)}")
        inputs.append(vec)
    inputs_np = np.array(inputs)
    return inputs_np, time_steps, input_dim


inputs, time_steps, input_dim = user_input()

# Define output dimension (can be different from hidden_dim)
output_dim = 2

rnn = VanillaRNN(input_dim=input_dim, hidden_dim=5, output_dim=output_dim)
hidden_states, outputs = rnn.forward(inputs)

print("RNN hidden states and outputs at each time step:")
for t in range(time_steps):
    print(f"Time step {t + 1}:")
    print(f"  Hidden state h: {hidden_states[t]}")
    print(f"  Output y: {outputs[t]}")
