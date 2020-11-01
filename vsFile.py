import numpy as np
#from numpy import exp, dot, random, array
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
NUM_EXAMPLES = 5620
DIMENSIONS = [65, 10, 10]


#read in and store data

class Network(object):
    def __init__(self):
        self.num_layers = len(DIMENSIONS)
        self.nodes = [] #array of 85 nodes with 4 different types

    def initializeNodes(self):
        #create 1 bias node first
        self.nodes.append(Node("bias"))
        for i in DIMENSIONS:
            #k is to reference node type
            k = 0
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

    def initialize_weights(self):
        for node in self.nodes:
            for connection in node.connections: #may not correctly reference node
                connection.setValue(random.uniform(low=-0.15, high=0.15,))

            
        # self.weights
        # self.biases
#create network class
    #decide dimensions
    #initialize weights and biases as relationa values of node objects

class Node(object):
    def __init__(self, type):
        self.weights = dict()
        self.bias
        self.type = type
        self.connections
        
    def initializeConnections(self, node):
        if type == "input":
            for i in range(DIMENSIONS[1]):
                self.weights[connection] = 0
        if type == "middle":

        if type == "output":
            
        if type == "bias":



    #def intiialize connections
    #def setValue

#create node object class with:
    #weight to every node it is connected to
    #type: input, middle or output


#create example class
    #the data read in












# class Perceptron(object):

#     def __init__(self, sizes):

#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)]

#         # for x, y in zip(sizes[:-1], sizes[1:])]
#         # print(self.weights)

# synaptic_weights = random.uniform(low=-1, high=1, size=(3, 1))
#     return synaptic_weights
