import numpy as np

class Node(object):

    @classmethod
    def createNodeWithRandomWeights(cls, numberOfInputs, numberOfClasses, learningRate, bias):
        cls.inputs = np.random.rand(numberOfInputs + bias, 1)
        cls.numberOfClasses = numberOfClasses
        cls.learningRate = learningRate

        return cls


class Task1NeuralNetwork(object):
    @classmethod
    def getIrisDataFromTextFile(cls):
        file = open("../task_resources/iris_data.txt")

        allLines = file.readlines()

        file.close()

        testCases = allLines[1:]

        return testCases

    @classmethod
    def splitDatasetToFeaturesAndLables(cls, dataSet):
        labels = []
        features = []

        for dataSetEntry in dataSet:
            splitEntry = dataSetEntry.split(',')

            cls.turnListToFloat(splitEntry)

            features.append(splitEntry[0:4])
            labels.append(splitEntry[-1])

        return features, labels

    @classmethod
    def turnListToFloat(cls, splitEntry):
        for i in range(3):
            splitEntry[i] = float(splitEntry[i])

    @classmethod
    def createNode(cls, numberOfInputs, numberOfClasses, learningRate, bias):
        return Node.createNodeWithRandomWeights\
                    (numberOfInputs, numberOfClasses, learningRate, bias)

    @classmethod
    def setEpochs(cls, epochs):
        cls.epochs = epochs

    @classmethod
    def signumValue(cls, netValue):
        if netValue > 0:
            return 1
        else:
            return -1
