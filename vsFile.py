import numpy as np
import random
import math
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01 #learning rate
NUM_EPOCHS = 3 #number of EPOCHS
NUM_EXAMPLES = 5620 #number of input examples
TRAINING_GRAPH = [NUM_EPOCHS] #Keeps track of average error across epochs
TEST_GRAPH = [NUM_EPOCHS]
BAR_GRAPH_DATA = [10]

#Perceptron class 
class Network(object):

    def __init__(self):

        self.data = readData()
        self.testSet = []
        self.trainingSet = []
        self.weights = self.initializeWeights() #creates a 10 x 64 array of randomly initialized weights  
        #self.splitData(self.data)

    def initializeWeights(self):

        #create 10 x 64 matrix holding randomly generated weights
        #rows represent output nodes, each columb being an input node's data
        return np.random.uniform(low=-0.1, high=0.1, size=(10,65))

    def splitData(self, data, dataSplit):

        random.shuffle(data)
        split = int(NUM_EXAMPLES * dataSplit)
        self.trainingSet = data[:split]
        self.testSet = data[split+1:]
        #print(self.trainingSet)
        #print(self.testSet)


    def train(self, dataSplit):

        self.weights = self.initializeWeights() #reset weights for next data split
        self.splitData(self.data, dataSplit) #split data
        print("Traning on split: " + str(dataSplit))

        for epoch in range(NUM_EPOCHS): 
            errorSum = 0

            print("Epoch number: " + str(epoch + 1))
            for example in self.trainingSet:  
                outNodeNum = 0 #used to index output nodes
                errorVector = [] #array to keep track of error for each example

                for outputNode in self.weights: #iterating output nodes in weights array
                    weightedSum = 0 #Keeps track of the weights sum of all node inputs
                    inputNodeNum = 0 #keeps track of which node in data to index

                    for inputNodeWeight in outputNode: #for each weight going to an ouput node
                        if inputNodeNum == 64: #check if weight coresponds to bias node
                            weightedSum += inputNodeWeight #only add the weight (input = 1)
                        else:
                            weightedSum += inputNodeWeight * example.inputData[inputNodeNum]

                        inputNodeNum += 1
                        
                    output = self.sigmoid(weightedSum) #pass weighted sum through activatoin function
                    error = example.expectedOutput[outNodeNum] - output #error = difference between target and output
                    errorVector.append(error) #add error to array

                    adjustmentNode = 0 #Used to index which node's weight is being adjusted
                    for inputNodeWeight in outputNode:

                        if adjustmentNode == 64: #If bias node:
                            #weight adjustment doesn't account for input as it equals 1
                             weightAdjustment = self.sigmoidDerivitive(output) * error * LEARNING_RATE
                        else:
                            weightAdjustment = self.sigmoidDerivitive(output) * error * example.inputData[adjustmentNode] * LEARNING_RATE
                        
                        #adjust weights
                        self.weights[outNodeNum][adjustmentNode] += weightAdjustment
                        adjustmentNode += 1
                    
                    #increase index of output node
                    outNodeNum += 1
                #break
                #calculates Euclidian distance between target outputs and net output 
                correctness = eucDistance(errorVector)
                #add to average error then divide by length of training set to get average
                errorSum += correctness

            testError = self.test()
            averageError = errorSum/len(self.trainingSet)
            TRAINING_GRAPH.append(averageError)
            TEST_GRAPH.append(testError)
            #GRAPH_DATA.append([TRAINING_SPLIT, 0, epoch, averageError])
            print("Training Set Euclidian Distance: " + str(averageError))
            print("Test Set Euclidian Distance: " + str(testError))



    def test(self):

        #print("Starting Test Set")
        errorSum = 0

        for example in self.testSet:  
            outNodeNum = 0 #used to index output nodes
            errorVector = [] #array to keep track of error for each example

            for outputNode in self.weights: #iterating output nodes in weights array
                weightedSum = 0 #Keeps track of the weights sum of all node inputs
                inputNodeNum = 0 #keeps track of which node in data to index

                for inputNodeWeight in outputNode: #for each weight going to an ouput node
                    if inputNodeNum == 64: #check if weight coresponds to bias node
                        weightedSum += inputNodeWeight #only add the weight (input = 1)
                    else:
                        weightedSum += inputNodeWeight * example.inputData[inputNodeNum]

                    inputNodeNum += 1
                    
                output = self.sigmoid(weightedSum) #pass weighted sum through activatoin function
                error = example.expectedOutput[outNodeNum] - output #error = difference between target and output
                errorVector.append(error) #add error to array
                
                #increase index of output node
                outNodeNum += 1
            #break
            #calculates Euclidian distance between target outputs and net output 
            correctness = eucDistance(errorVector)
            #add to average error then divide by length of training set to get average
            errorSum += correctness

        averageError = errorSum/len(self.testSet)
        BAR_GRAPH_DATA.append(averageError)
        #GRAPH_DATA.append([TRAINING_SPLIT, 0, epoch, averageError])
        return averageError

    #Function to calculate sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Function to calculate sigmoid derivitive
    def sigmoidDerivitive(self, x):
        return x * (1 - x)

#Class to hold example objects
#Takes an array of node input values and an array of values representing the target output
class Example(object):

    def __init__(self, inputData, expectedOutput):

        self.inputData = inputData
        self.expectedOutput = expectedOutput

#Function used to calculate Euclidean distance between two vectors
#Note that this function takes a list where each value already represents 
#the distance between target and output
def eucDistance(list1):

    sqrdErrors = [i ** 2 for i in list1] #square all values
    totalVal = 0

    for val in sqrdErrors:
        totalVal += val #Sum all squared values

    return math.sqrt(totalVal) #return square root
     


def readData():

    #open the file
    file = open('digit-examples-all.txt', 'r') 
    lines = file.readlines() #read the lines

    #initialize arrays to hold input and output data
    numDataPoints = len(lines)//2
    bitMaps = []
    numAnswers = []
    data = []

    #read data into arrays
    for i in range(len(lines)):
        if i % 2 == 0:
            bitMaps.append(lines[i].split()[1:-1])
        else:
            numAnswers.append(lines[i].split()[1:-1])

    #convert from strings to floats
    for i in range(numDataPoints):

        floatMap = [float(ele) for ele in bitMaps[i]]
        floatAns = [float(ele) for ele in numAnswers[i]]
        #append example objects holding input and output data to data array
        data.append(Example(floatMap, floatAns))

    return data

#Function to graph Net data
def graphEpochs(trainingData, testData, trainingSplit):

    plt.plot(trainingData)
    plt.plot(testData)
    plt.title("Training Split: " + str(trainingSplit))
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Epoch Number")
    plt.xlim(1, NUM_EPOCHS)
    plt.ylim(0, 1.5)
    plt.show()
    

def bargraph(errorData, trainingSplits):

    fig = plt.figure()
    graph = fig.add_axes([0,0,1,1])
    graph.bar(trainingSplits, errorData)
    plt.show()

if __name__ == "__main__":

    net = Network()
    trainingSplits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #print(net.examples[0].expectedOutput)
    #print(net.examples[0].inputData)
    #print(net.weights)
    #net.train(0.7)
    #graphEpochs(TRAINING_GRAPH, TEST_GRAPH, 0.7)

    for split in trainingSplits:
        net.train(split)
    
    bargraph(BAR_GRAPH_DATA, trainingSplits)
        #graphEpochs(TRAINING_GRAPH, TEST_GRAPH, split)

  


    


