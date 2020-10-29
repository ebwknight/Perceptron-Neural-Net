from numpy import exp, dot, random, array

"""Python code for simple Artificial Neural Network with one hidden layer"""

class Perceptron(object):

    def __init__(self):

        self.weights = random.uniform(low=-0.15, high=0.15, size=(3, 2))
        self.num_layers = 3
        #self.biases = random.uniform(size=(2))
        self.epochs = 1000
        self.lr = 0.01


    def feedForward(self, inputs):

        for weight, bias in zip(self.biases, self.weights):
            output = sigmoid(dot(inputs, weight) + bias)
        return output

    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)


    def train(inputs, expected_output):
        for epoch in range(self.epochs):
        # Forward pass -- Pass the training set through the network.
            predicted_output = feedForward(inputs)

        # Backaward pass
        # Calculate the error
            error = sigmoid_derivative(predicted_output) * (expected_output - predicted_output)

        # Adjust the weights and bias by a factor
            for w in (self.weights):
                weight_factor = dot(inputs.T, error) * learning_rate
            #bias_factor = error * learning_rate

        # Update the synaptic weights
                w += weight_factor

        # Update the bias
            #self.bias += bias_factor

            if ((epoch % 1000) == 0):
                print("Epoch", epoch)
                print("Predicted Output = ", predicted_output.T)
                print("Expected Output = ", expected_output.T)
                print()
                
        return self.weights





if __name__ == "__main__":
    # Initialize random weights for the network
    weights = initialize_weights()

    # The training set
    inputs = array([[0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1]])

    # Target set
    expected_output = array([[1, 0, 1]]).T

    # Test set
    test = array([1, 0, 1])

    # Train the neural network
    trained_weights = train(inputs, expected_output, weights, bias=0.001, learning_rate=0.98,
                            training_iterations=1000000)

    # Test the neural network with a test example
    accuracy = (learn(test, trained_weights, bias=0.01)) * 100

    print("accuracy =", accuracy[0], "%")