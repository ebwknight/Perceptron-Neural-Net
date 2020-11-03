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

        self.examples = [] #array of Example objects storin ginput and output data
        self.weights = self.initializeWeights() #creates a 10 x 64 array of randomly initialized weights

        self.readData() #reads in data and populates self.examples
  

    def initializeWeights(self):

        #create 10 x 64 matrix holding randomly generated weights
        #rows represent output nodes, each columb being an input node's data
        return np.random.uniform(low=-0.15, high=0.15, size=(10,64))

    """
    def feedForward(self, inputData):
        #print(self.sigmoid(np.dot(self.weights, inputData)))
        output = [] #array to hold nets output data
        
        for outputNode in self.weights: 
            weightedSum = 0 
            inputNode = 0 #keeps track of which node in data to index
            #print(outputNode)

            for inputNodeWeight in outputNode:
                #print("inputNodeWeight: " + inputNodeWeight)
                #print("inputData[inputNode]: " + inputData[inputNode])
                weightedSum += inputNodeWeight * inputData[inputNode]
                inputNode += 1

            output.append(self.sigmoid(weightedSum))

        return output
    """

    def train(self):

        for epoch in range(2):

            print("Epoch number: " + str(epoch))
            for example in self.examples:  
                outNodeNum = 0

                for outputNode in self.weights: 
                    weightedSum = 0 
                    inputNodeNum = 0 #keeps track of which node in data to index

                    for inputNodeWeight in outputNode:
                        weightedSum += inputNodeWeight * example.inputData[inputNodeNum]
                        
                        output = self.sigmoid(weightedSum)
                        error = example.expectedOutput[outNodeNum] - output
                        weightAdjustment = self.sigmoidDerivitive(output) * error * example.inputData[inputNodeNum] * LEARNING_RATE
                        self.weights[outNodeNum][inputNodeNum] += weightAdjustment

                        inputNodeNum += 1
                    outNodeNum += 1
  

        

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

  


    


