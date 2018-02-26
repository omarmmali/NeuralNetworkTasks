import unittest
import numpy as np
from sample.task1 import Task1NeuralNetwork

class TestNeuralTask1(unittest.TestCase):
    def testLoadingIrisDataSetFromTextFile(self):
        dataSet = Task1NeuralNetwork.getIrisDataFromTextFile()

        self.assertEqual(150, len(dataSet))
        self.assertEqual('5.1,3.5,1.4,0.2,Iris-setosa', dataSet[0].rstrip())
        self.assertEqual('5.9,3.0,5.1,1.8,Iris-virginica', dataSet[-1].rstrip())

    def testSplittingDataToFeaturesAndLables(self):
        dataSet = ['1,2,3,4,koko','5.0,6,7,8,koko2']

        features, lables = Task1NeuralNetwork.splitDatasetToFeaturesAndLables(dataSet)

        self.assertEqual(4, len(features[0]))
        self.assertEqual(5, features[1][0])

        self.assertEqual('koko2', lables[1])
        self.assertEqual(2, len(lables))

    def testCreatingANodeWithSpecifiedNumberOfInputsWithoutBias(self):
        numberOfInputs = 2
        bias = 0
        learningRate = 0.1
        numberOfClasses = 2

        node = Task1NeuralNetwork.createNode(numberOfInputs, learningRate, bias)

        self.assertEqual(numberOfInputs, len(node.weights))
        self.assertEqual(learningRate, node.learningRate)
        self.assertEqual(numberOfInputs, node.numberOfInputs)
        self.assertEqual(0, node.bias)

    def testCreatingANodeWithSpecifiedNumberOfInputsWithBias(self):
        numberOfInputs = 2
        bias = 1
        learningRate = 0.1
        numberOfClasses = 2

        node = Task1NeuralNetwork.createNode(numberOfInputs, learningRate, bias)

        self.assertEqual(numberOfInputs, len(node.weights) - 1)
        self.assertEqual(learningRate, node.learningRate)
        self.assertEqual(numberOfInputs, node.numberOfInputs)
        self.assertEqual(1, node.bias)

    def testSettingEpochs(self):
        epochs = 2

        Task1NeuralNetwork.setEpochs(epochs)

        self.assertEqual(epochs, Task1NeuralNetwork.epochs)

    def testSettingInputsInNode(self):
        features = [1, 2, 3, 4]

        dummyNode = Task1NeuralNetwork.createNode(4, 0.1, 1)

        dummyNode.setInputValues(features)

        for i in range(dummyNode.numberOfInputs):
            self.assertEqual(features[i], dummyNode.inputs[i])

    def testSignumReturnsNegativeOneForNegativeValues(self):
        netValue = -0.56

        output = Task1NeuralNetwork.signumValue(netValue)

        self.assertEqual(-1, output)

    def testSignumReturnsOneForPositiveValues(self):
        netValue = 0.344

        output = Task1NeuralNetwork.signumValue(netValue)

        self.assertEqual(1, output)

    def testNetValueWithBias(self):
        dummyNode = Task1NeuralNetwork.createNode(2, 0.1, 1)

        dummyWeights = [1, 1, 1]
        dummyNode.weights = dummyWeights

        output = dummyNode.calculateNetValue([1, 2])

        self.assertEqual(4, output)

    def testNetValueWithoutBias(self):
        dummyNode = Task1NeuralNetwork.createNode(2, 0.1, 0)

        dummyWeights = [1, 1]
        dummyNode.weights = dummyWeights

        output = dummyNode.calculateNetValue([1, 1])

        self.assertEqual(2, output)


    def testUpdatingWeightsOfNode(self):
        dummyNode = Task1NeuralNetwork.createNode(2, 4, 0)

        error = 0.5

        dummyNode.weights[0] = 1
        dummyNode.weights[1] = 1

        dummyNode.updateWeights(error, [1, 1])

        self.assertEqual(3, dummyNode.weights[0])