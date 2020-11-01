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
        self.nodes #array of 85 nodes with 4 different types
        self.connections #dictionary with node touples as key and weights as values

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

    def initializeConnections(self):

        for startNode in self.nodes:

            if startNode.type == "bias":

                for endNode in self.nodes:

                    if endNode.type == "output":

                        if not self.connections.contains(startNode, endNode):
                            self.connections[startNode, endNode] = random.uniform(low=-0.15, high=0.15)

                        
            if startNode.type == "input":

                for endNode in self.nodes:

                    if endNode.type == "middle":

                        if not self.connections.contains(startNode, endNode):
                            self.connections[startNode, endNode] = random.uniform(low=-0.15, high=0.15)


            if startNode.type == "output":

                for endNode in self.nodes:

                    if endNode.type == "output":

                        if not self.connections.contains(startNode, endNode):
                            self.connections[startNode, endNode] = random.uniform(low=-0.15, high=0.15)
                     

            if startNode.type == "middle":

                for endNode in self.nodes:

                    if endNode.type == "input" or "output":

                        if not self.connections.contains(startNode, endNode):
                            self.connections[startNode, endNode] = random.uniform(low=-0.15, high=0.15)

                        
"""
    def initialize_weights(self):
        for node in self.nodes:
            for connection in node.connections: #may not correctly reference node
                connection.setValue(random.uniform(low=-0.15, high=0.15,))
"""
            
        # self.weights
        # self.biases
#create network class
    #decide dimensions
    #initialize weights and biases as relationa values of node objects

class Node(object):

    def __init__(self, type):
 
        self.bias
        self.type = type

"""

class biasNode(Node):

    def __init__(self, bias):

        self.bias = bias
"""

def main():

    net = Network()

    print(net.nodes) 

    
if __name__ == "__main__":
    main()

    


