import numpy as np
"""
here is da finished code man (dis took me 2 hrs)
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        """
        input_size: int, number of input neurons
        hidden_layers: list of ints, number of neurons in each hidden layer
        output_size: int, number of output neurons
        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases randomly
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i+1], self.layers[i])
            bias = np.random.randn(self.layers[i+1], 1)
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        """
        x: input vector (list or numpy array) of shape (input_size,)
        returns: output vector of shape (output_size,)
        """
        a = np.array(x).reshape(-1, 1)  # column vector
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a.flatten()
    
    def add_input(self, new_inputs):
        """
        Add new inputs to the network by increasing the input layer size.
        new_inputs: int and number of neurons to add to da inputt layerr
        """
        # Update input layer size
        old_input_size = self.layers[0]
        new_input_size = old_input_size + new_inputs
        self.layers[0] = new_input_size
        
        # Adjust weights of first layer to accommodate new inputs
        old_weights = self.weights[0]
        old_biases = self.biases[0]
        
        # Initialize new weights for the new inputs
        new_weights_part = np.random.randn(old_weights.shape[0], new_inputs)
        
        # Concatenate old weights with new weights horizontally
        new_weights = np.hstack((old_weights, new_weights_part))
        
        self.weights[0] = new_weights
        self.biases[0] = old_biases  # biases remain the same

# Example usage:
if __name__ == "__main__":
    # Create a network with 3 inputs, two hidden layers (4 and 5 neurons), and 2 outputs
    nn = NeuralNetwork(input_size=3, hidden_layers=[4,5,67], output_size=1)
    
    # Input vector
    input_vector = [0.5, 0.1, 0.9]
    
    # Forward pass
    output = nn.forward(input_vector)
    print("Output:", output)
    
    # Add 2 more inputs
    nn.add_input(2)
    
    # New input vector with 5 inputs
    new_input_vector = [0.5, 0.1, 0.9, 0.3, 0.7]
    new_output = nn.forward(new_input_vector)
    print("Output after adding inputs:", new_output)
