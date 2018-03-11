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
    def calculateNetValue(cls, inputs,f1_index ,f2_index):
        netValue = 0
        netValue += float(inputs[f1_index]) * cls.weights[0]
        netValue += float(inputs[f2_index]) * cls.weights[1]

        if cls.bias:
            netValue += cls.weights[2]

        return netValue

    @classmethod
    def updateWeights(cls, error, inputs,f1_index,f2_index):
        changeInWeights = float(cls.learningRate) * float(error)
        cls.weights[0] += float(changeInWeights) * float(inputs[f1_index])
        cls.weights[1] += float(changeInWeights) * float(inputs[f2_index])


        if cls.bias:
            cls.weights[2] += changeInWeights

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
    def fit(cls, classes, bias, learningRate, epochs,f1_index,f2_index):
        dataSet = cls.getIrisDataFromTextFile()
        features, lables = cls.splitDatasetToFeaturesAndLables(dataSet)
        cls.setEpochs(epochs)
        cls.curNode = Node.createNodeWithRandomWeights(2, learningRate, bias)

        firstClassDataSet, secondClassDataSet = cls.getClassesDataSets(classes, features, lables)

        testSet, trainingSet = cls.prepareTrainingAndTestSets(firstClassDataSet, secondClassDataSet)

        for trainingIteration in range(cls.epochs):

            cls.executeTrainingStep(trainingSet,f1_index,f2_index)

            errors = cls.executeCrossValidationStep(testSet,f1_index,f2_index)

            if errors == 0:
                cls.accuracy = 1
                return
            else:
                cls.accuracy = 1 - (errors / len(testSet))

    @classmethod
    def executeCrossValidationStep(cls, testSet,f1_index,f2_index):
        errors = 0
        for features, label in testSet:
            prediction = cls.signumValue(cls.curNode.calculateNetValue(features,f1_index,f2_index))
            curError = label - prediction
            if curError != 0:
                errors += 1
        return errors

    @classmethod
    def executeTrainingStep(cls, trainingSet,f1_index,f2_index):
        for features, label in trainingSet:
            prediction = cls.signumValue(cls.curNode.calculateNetValue(features,f1_index,f2_index))
            error = label - prediction
            cls.curNode.updateWeights(error, features,f1_index,f2_index)

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
            if lables[i].rstrip() == classes[0]:
                firstClassDataSet.append((features[i], -1))
            elif lables[i].rstrip() == classes[1]:
                secondClassDataSet.append((features[i], 1))
        return firstClassDataSet, secondClassDataSet
    @classmethod
    def predict(cls,features):
        print(features,cls.curNode.weights)
        netValue = float(features[0]) * cls.curNode.weights[0] + float(features[1]) * cls.curNode.weights[1]
        prediction = cls.signumValue(netValue)
        return prediction, netValue