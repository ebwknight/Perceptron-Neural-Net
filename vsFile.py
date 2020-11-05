import numpy as np
import random
import math
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01 #learning rate
NUM_EPOCHS = 50 #number of EPOCHS
NUM_EXAMPLES = 5620 #number of input examples
TRAINING_GRAPH = [] #Keeps track of average error across epochs
TEST_GRAPH = [] #Holds acerage error per epoch
BAR_GRAPH_DATA = [] #array to hold average error per test data split

#Perceptron class that reads hand-written digits
class Network(object):

    def __init__(self):

        self.data = readData() #holds an array of Exmaple objects containing input and target output data
        self.testSet = [] #holds test set example objects
        self.trainingSet = [] #holds trning set example objects
        self.weights = self.initializeWeights() #creates a 10 x 65 array of randomly initialized weights  


    def initializeWeights(self):
        #create 10 x 65 matrix holding randomly generated weights
        #rows represent output nodes, each columb being an input node's data
        return np.random.uniform(low=-0.1, high=0.1, size=(10,65))

    #Takes read-in data and a desired data split
    #sorts data into the training set and test set arrays
    def splitData(self, data, dataSplit):

        random.shuffle(data) #shuffle all the data
        #split relative to data size
        split = int(NUM_EXAMPLES * dataSplit)
        self.trainingSet = data[:split]
        self.testSet = data[split+1:]


    #Main training function that takes a desired data split
    def train(self, dataSplit):

        self.weights = self.initializeWeights() #reset weights for next data split
        self.splitData(self.data, dataSplit) #split data
        print("Traning on split: " + str(dataSplit))

        for epoch in range(NUM_EPOCHS): 
            errorSum = 0 #used to keep track of average error

            print("Epoch number: " + str(epoch + 1))
            for example in self.trainingSet: #for example in training set:
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

            testError = self.test() #try traing data against current weights and return average error of test set
            #training set average error
            averageError = errorSum/len(self.trainingSet)
            #Save these two values for graphing later
            TRAINING_GRAPH.append(averageError)
            TEST_GRAPH.append(testError)

            #If on the last epoch, append test set correctness to the bar graph data
            if epoch == (NUM_EPOCHS - 1):
                BAR_GRAPH_DATA.append(testError)

            print("Training Set Euclidian Distance: " + str(averageError))
            print("Test Set Euclidian Distance: " + str(testError))


    #Function that feeds test set data through the net without adjusting weights
    #returns the average error of the test set
    def test(self):

        errorSum = 0 #Keeps track of sum of errors

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

                #No weight adjustment

            #calculates Euclidian distance between target outputs and net output 
            correctness = eucDistance(errorVector)
            #add to average error then divide by length of training set to get average
            errorSum += correctness

        averageError = errorSum/len(self.testSet) #average a
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

#Function to graph training set error vs test set error per epoch
#should guage overfitting and underfitting
def graphEpochs(trainingData, testData, trainingSplit):

    plt.plot(trainingData)
    plt.plot(testData)
    plt.title("Training Split: " + str(trainingSplit))
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Epoch Number")
    plt.xlim(1, NUM_EPOCHS)
    plt.ylim(0, 1.5)
    plt.show()
    

def graph(trainingSplits, errorData):

    plt.plot(trainingSplits, errorData)
    plt.title("Training Split vs Test Error")
    plt.xlabel("Training Split")
    plt.ylabel("Test Error")
    plt.show()

if __name__ == "__main__":

    #lines to test graph functionality
    # y = [0.9002910468679929, 0.8180141685100354, 0.7431281297271973, 0.7009388577523336, 0.6612819516465702, 0.6310570119479071, 0.5972628364656487, 0.5793497428478508, 0.5578205773847719]
    # x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # graph(x, y)

    net = Network()
    trainingSplits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    for split in trainingSplits:
        net.train(split)
    print(BAR_GRAPH_DATA)
    print(trainingSplits)
    graph(trainingSplits, BAR_GRAPH_DATA)
        #graphEpochs(TRAINING_GRAPH, TEST_GRAPH, split)

  


    


