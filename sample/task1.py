import numpy as np
import random

class Node(object):

    @classmethod
    def createNodeWithRandomWeights(cls, numberOfInputs, learningRate, bias):
        cls.weights = np.random.rand(numberOfInputs + bias, 1)
        cls.numberOfInputs = numberOfInputs
        cls.learningRate = learningRate
        cls.bias = bias

        return cls

    @classmethod
    def setInputValues(cls, features):
        if cls.bias:
            features.append(1)

        cls.inputs = np.array(features)

    @classmethod
    def calculateNetValue(cls, inputs):
        netValue = 0

        for i in range(cls.numberOfInputs):
            netValue += inputs[i] * cls.weights[i]

        if cls.bias:
            netValue += cls.weights[-1]

        return netValue

    @classmethod
    def updateWeights(cls, error, inputs):
        changeInWeights = cls.learningRate * error

        for i in range(len(inputs)):
            cls.weights[i] += changeInWeights * inputs[i]

        if cls.bias:
            cls.weights[-1] += changeInWeights

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
    def createNode(cls, numberOfInputs, learningRate, bias):
        return Node.createNodeWithRandomWeights(numberOfInputs, learningRate, bias)

    @classmethod
    def setEpochs(cls, epochs):
        cls.epochs = epochs

    @classmethod
    def signumValue(cls, netValue):
        if netValue >= 0:
            return 1
        else:
            return -1

    @classmethod
    def fit(cls, features, classes, bias, learningRate):
        dataSet = cls.getIrisDataFromTextFile()
        features, lables = cls.splitDatasetToFeaturesAndLables(dataSet)

        cls.curNode = Node.createNodeWithRandomWeights(len(features), learningRate, bias)

        firstClassDataSet, secondClassDataSet = cls.getClassesDataSets(classes, features, lables)

        testSet, trainingSet = cls.prepareTrainingAndTestSets(firstClassDataSet, secondClassDataSet)

        for trainingIteration in range(cls.epochs):

            cls.executeTrainingStep(trainingSet)

            errors = cls.executeCrossValidationStep(testSet)

            if errors == 0:
                cls.accuracy = 1
                return
            else:
                cls.accuracy = 1 - (errors / len(testSet))

    @classmethod
    def executeCrossValidationStep(cls, testSet):
        errors = 0
        for features, label in testSet:
            prediction = cls.signumValue(cls.curNode.calculateNetValue(features))
            curError = label - prediction
            if curError != 0:
                errors += 1
        return errors

    @classmethod
    def executeTrainingStep(cls, trainingSet):
        for features, label in trainingSet:
            prediction = cls.signumValue(cls.curNode.calculateNetValue(features))
            error = label - prediction
            cls.curNode.updateWeights(error, features)

    @classmethod
    def prepareTrainingAndTestSets(cls, firstClassDataSet, secondClassDataSet):
        random.shuffle(firstClassDataSet)
        random.shuffle(secondClassDataSet)
        firstClassTrainingSet = firstClassDataSet[0:30]
        secondClassTrainingSet = secondClassDataSet[0:30]
        firstClassTestSet = firstClassDataSet[30:]
        secondClassTestSet = secondClassDataSet[30:]
        trainingSet = firstClassTrainingSet + secondClassTrainingSet
        testSet = firstClassTestSet + secondClassTestSet
        return testSet, trainingSet

    @classmethod
    def getClassesDataSets(cls, classes, features, lables):
        firstClassDataSet = []
        secondClassDataSet = []
        for i in range(len(lables)):
            if lables[i] == classes[0]:
                firstClassDataSet.append((features[i], -1))
            elif lables[i] == classes[1]:
                secondClassDataSet.append((features[i], 1))
        return firstClassDataSet, secondClassDataSet

    def predict(cls, features):
        prediction = cls.signumValue(cls.curNode.calculateNetValue(features))
        return prediction