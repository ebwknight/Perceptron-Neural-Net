import numpy as np
import math
#from numpy import exp, dot, random, array
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
NUM_EXAMPLES = 5620
DIMENSIONS = [64, 10]


#read in and store data

class Network(object):

    def __init__(self):

        self.examples = []
        self.weights = self.initializeWeights()

        self.readData()
    """
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
                    #create j output nodes
                    self.nodes.append(Node("output"))
            k = k + 1
    """
    """
    def initializeWeights(self):
        
        for startNode in self.nodes:
            if startNode.type == "bias":
                for endNode in self.nodes:
                    if endNode.type == "output":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)
                        
            if startNode.type == "input":
                for endNode in self.nodes:
                    if endNode.type == "output":
                        if (startNode, endNode) not in self.weights:
                            self.weights[startNode, endNode] = np.random.uniform(low=-0.15, high=0.15)

    """

    def initializeWeights(self):

        #create 64 x 10 matrix holding randomly generated weights
        return np.random.uniform(low=-0.15, high=0.15, size=(10,64))



    def feedForward(self, inputData):
        #print(self.sigmoid(np.dot(self.weights, inputData)))
        output = []
        
        for outputNode in self.weights:
            weightedSum = 0
            nodeInput = 0
            #print(outputNode)

            for inputNodeWeight in outputNode:
                #print("inputNodeWeight: " + inputNodeWeight)
                #print("inputData[nodeInput]: " + inputData[nodeInput])
                weightedSum += inputNodeWeight * inputData[nodeInput]
                nodeInput += 1

            output.append(self.sigmoid(weightedSum))
            print(output)
            break

        return output


    def train(self):

        for epoch in range(NUM_EPOCHS):

            for example in self.examples:

                #returns an array of 10 numbers representing 
                netOutput = self.feedForward(example.inputData)
                print(netOutput)
                break


                #error = self.sigmoidDerivitive(netOutput) * (example.expectedOutput - netOutput)

                #weightAdjustment = np.dot(examples.inputData, error) * LEARNING_RATE
                #self.weights += weightAdjustment

        return self.weights

    def readData(self):

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


        for i in range(numDataPoints):

            floatMap = [float(ele) for ele in bitMaps[i]]
            floatAns = [float(ele) for ele in numAnswers[i]]
            self.examples.append(Example(floatMap, floatAns))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoidDerivitive(self, x):
        return x * (1 - x)
        #print(input_data[0])
                     

"""
class Node(object):

    def __init__(self, type):
 
        #self.bias
        self.type = type
"""
class Example(object):

    def __init__(self, inputData, expectedOutput):

        self.inputData = inputData
        self.expectedOutput = expectedOutput

    def __repr__(self):

        return (self.inputData, self.expectedOutput)



if __name__ == "__main__":

    net = Network()
    #print(net.examples[0].expectedOutput)
    #print(net.examples[0].inputData)
    net.train()

    




    


