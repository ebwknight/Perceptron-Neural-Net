import numpy as np
import random
import math
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
NUM_EXAMPLES = 5620
DIMENSIONS = [64, 10]




class Network(object):

    def __init__(self):

        self.data = readData()
        self.testSet = []
        self.trainingSet = []
        self.weights = self.initializeWeights() #creates a 10 x 64 array of randomly initialized weights  
        self.splitData(self.data)

    def initializeWeights(self):

        #create 10 x 64 matrix holding randomly generated weights
        #rows represent output nodes, each columb being an input node's data
        return np.random.uniform(low=-0.15, high=0.15, size=(10,64))

    def splitData(self, data):

        random.shuffle(data)
        split = int(NUM_EXAMPLES * 0.8)
        self.trainingSet = data[:split]
        self.testSet = data[split+1:]


    def train(self):

        for epoch in range(5): 
            averageError = 0


            print("Epoch number: " + str(epoch))
            for example in self.trainingSet:  
                outNodeNum = 0
                errorVector = []

                for outputNode in self.weights: 
                    weightedSum = 0 
                    inputNodeNum = 0 #keeps track of which node in data to index

                    for inputNodeWeight in outputNode:
                        weightedSum += inputNodeWeight * example.inputData[inputNodeNum]
                        inputNodeNum += 1

                    output = self.sigmoid(weightedSum)
                    error = example.expectedOutput[outNodeNum] - output
                    errorVector.append(error)

                    adjustmentNode = 0
                    for inputNodeWeight in outputNode:
                        #print("output: " + str(output))
                        #print(error)
                        #print(example.inputData[adjustmentNode])

                        weightAdjustment = self.sigmoidDerivitive(output) * error * example.inputData[adjustmentNode] * LEARNING_RATE
                        print(weightAdjustment)
                        #break
                        self.weights[outNodeNum][adjustmentNode] += weightAdjustment
                    #break
                #break
                outNodeNum += 1
                correctness = eucDistance(errorVector)
                averageError += correctness
            print("Error: " + str(averageError/len(self.trainingSet)))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoidDerivitive(self, x):
        return x * (1 - x)


class Example(object):

    def __init__(self, inputData, expectedOutput):

        self.inputData = inputData
        self.expectedOutput = expectedOutput


def eucDistance(list1):

    sqrdErrors = [i ** 2 for i in list1]
    totalVal = 0

    for val in sqrdErrors:
        totalVal += val

    return math.sqrt(totalVal)
     


def readData():

    file = open('digit-examples-all.txt', 'r') 
    lines = file.readlines()

    numDataPoints = len(lines)//2
    bitMaps = []
    numAnswers = []
    data = []

    for i in range(len(lines)):
        if i % 2 == 0:
            bitMaps.append(lines[i].split()[1:-1])
        else:
            numAnswers.append(lines[i].split()[1:-1])


    for i in range(numDataPoints):

        floatMap = [float(ele) for ele in bitMaps[i]]
        floatAns = [float(ele) for ele in numAnswers[i]]
        data.append(Example(floatMap, floatAns))

    return data


if __name__ == "__main__":

    net = Network()
    #print(net.examples[0].expectedOutput)
    #print(net.examples[0].inputData)
    net.train()

  


    


