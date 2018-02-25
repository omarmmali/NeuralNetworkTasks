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