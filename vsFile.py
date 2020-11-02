import numpy as np
#from numpy import exp, dot, random, array
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
NUM_EXAMPLES = 5620
DIMENSIONS = [64, 10, 10]


#read in and store data

class Network(object):

    def __init__(self):

        self.num_layers = len(DIMENSIONS)
        self.nodes = []#array of 85 nodes with 4 different types
        self.weights = {}#dictionary with node touples as key and weights as values
        self.data = readData()

    def initializeNodes(self):

        #create 1 bias node first
        #self.nodes.append(Node("bias"))
        k = 0
        for i in DIMENSIONS:
            #k is to reference node type
            for j in range(i):
                #create j nodes of type k
                if k == 0:
                    self.nodes.append(Node("input"))
                    #create j input nodes
                if k == 1:
                    #create j middle nodes
                    self.nodes.append(Node("middle"))
                if k == 2:
                    #create j output nodes
                    self.nodes.append(Node("output"))
            k = k + 1


    def initializeWeights(self):
        
        for startNode in self.nodes:
            """if startNode.type == "bias":
                for endNode in self.nodes:
                    if endNode.type == "output":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)
"""
                        
            if startNode.type == "input":
                for endNode in self.nodes:
                    if endNode.type == "middle":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)


            if startNode.type == "output":
                for endNode in self.nodes:
                    if endNode.type == "output":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)
                     

            if startNode.type == "middle":
                for endNode in self.nodes:
                    if endNode.type == "input" or "output":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)


    def feedForward(self, example):

        heavy = convert_to_array(self.weights)
        print(heavy)
        return self.sigmoid(np.dot(heavy, example.inputData))


    def train(self):

        """
        biasWeights = []
        otherWeights = []

        for weight in self.weights:

            if weight.key()[0].type == "bias":

                biasWeights.append(weight.value())

            else otherWeights.append(weight.value())
        """

        for epoch in range(NUM_EPOCHS):

            example = Example(self.data[epoch][0], self.data[epoch][1][0])
            # Forward pass -- Pass the training set through the network.
            networkOutput = self.feedForward(example)

            # Backaward pass
            # Calculate the error
            error = self.sigmoid_derivative(networkOutput) * (example.correctAnswer - networkOutput)

            # Adjust the weights and bias by a factor
            weightAdjustment = dot(example.inputData, error) * LEARNING_RATE
            #bias_factor = error * learning_rate

            # Update the synaptic weights
            self.weights += weightFactor

            # Update the bias
            #bias += bias_factor

            if ((epoch % 1000) == 0):
                print("Epoch", epoch)
                print("Predicted Output = ", netOutput.T)
                print("Expected Output = ", correctAnswer.T)
                print()

        return self.weights
        

    def sigmoid(x):
        return 1 / (1 + exp(-x))


    def sigmoid_derivative(x):
        return x * (1 - x)
        #print(input_data[0])
                     
"""
    def initialize_weights(self):
        for node in self.nodes:
            for connection in node.weights: #may not correctly reference node
                connection.setValue(random.uniform(low=-0.15, high=0.15,))
"""
            
        # self.weights
        # self.biases
#create network class
    #decide dimensions
    #initialize weights and biases as relationa values of node objects

class Node(object):

    def __init__(self, type):
 
        #self.bias
        self.type = type

class Example(object):

    def __init__(self, inputData, expectedOutput):

        self.inputData = inputData
        self.expectedOutput = expectedOutput

def convert_to_array(dictionary):
    '''Converts lists of values in a dictionary to numpy arrays'''
    stuff = dictionary.values()
    newStuff = list(stuff)
    return np.array(newStuff)

def readData():

        file = open('digit-examples-all.txt', 'r') 
        lines = file.readlines()

        numDataPoints = len(lines)//2
        input_data = np.empty(shape=(numDataPoints ,2, 64))

        bitMaps = []
        numAnswers = []

        for i in range(len(lines)):
            if i % 2 == 0:
                bitMaps.append(lines[i].split()[1:-1])
            else:
                numAnswers.append(lines[i].split()[1:-1])

        singleNumAnswers = []

        for i in range(len(numAnswers)):
            for j in range(len(numAnswers[i])):
                if numAnswers[i][j] == '1.0':
                    singleNumAnswers.append(j)

        for i in range(numDataPoints):
            input_data[i][0] = bitMaps[i]
            input_data[i][1] = singleNumAnswers[i]

        return input_data

"""

class biasNode(Node):

    def __init__(self, bias):

        self.bias = bias
"""
if __name__ == "__main__":

    net = Network()

    net.initializeNodes()
    net.initializeWeights()
    print(net.weights.values())
    net.train()




    


