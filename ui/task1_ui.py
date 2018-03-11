from sample import  task1
from tkinter import *
from tkinter import  ttk, StringVar, messagebox
import matplotlib.pyplot as plt
import numpy as np


class Task1Gui:
    def __init__(self, master):
        self.master = master
        self.createTexts()
        self.createButtons()
        self.createCombobox()
        self.dataSet = task1.Task1NeuralNetwork.getIrisDataFromTextFile()
        self.dataset_features, self.dataset_labels = task1.Task1NeuralNetwork.splitDatasetToFeaturesAndLables(self.dataSet)
        self.class1trainningset = self.dataset_features[0:30]
        self.class1testingset = self.dataset_features[30:50]
        self.class2trainningset = self.dataset_features[50:80]
        self.class2testingset = self.dataset_features[80:100]
        self.class3trainningset = self.dataset_features[100:130]
        self.class3testingset = self.dataset_features[130:150]

    def createButtons(self):
        # initialization
        self.close_button = Button(self.master, text="Close", command=self.master.quit, height = 2, width = 13)
        self.draw_button = Button(self.master, text="Draw", command=self.draw_dataset, height = 2, width = 13)
        self.train_button = Button(self.master, text="Train", command=self.start_trainning, height = 2, width = 13)
        self.test_button = Button(self.master, text="Test", command=self.test_features, height = 2, width = 16)

        # positions
        self.draw_button.place(relx=1, x=-600, y=230, anchor=NE)
        self.train_button.place(relx=1, x=-50, y=230, anchor=NE)
        self.close_button.place(relx=1, x= -15, y=500, anchor=NE)
        self.test_button.place(relx=1, x=-280, y=520, anchor=NE)


    def createTexts(self):
        # initialization
        self.label = Label(self.master, text="Iris-Data Neural Network",font=("Comic Sans", 22))
        self.classes = Label(self.master, text="Select Classes:-",font=("Comic Sans", 10))
        self.features = Label(self.master, text="Select Features :-",font=("Comic Sans", 10))
        self.epochs = Label(self.master, text="Number of epochs :-",font=("Comic Sans", 10))
        self.learningrate = Label(self.master, text= " Learning rate :-",font=("Comic Sans", 10))
        self.biaslabel = Label(self.master, text="Bias value",font=("Comic Sans", 10))
        self.stats = Label(self.master, text="Stats",font=("Comic Sans", 18))
        self.Accuracy = Label(self.master, text="Accuracy : ",font=("Comic Sans", 14))
        self.Accuracy_value = Label(self.master, text="",font=("Comic Sans", 14))
        self.Tests = Label(self.master, text="Test",font=("Comic Sans", 18))
        self.X1value = Label(self.master, text="X1 :",font=("Comic Sans", 14))
        self.X2value = Label(self.master, text="X2 :",font=("Comic Sans", 14))
        self.netValue = Label(self.master, text="Net Value :",font=("Comic Sans", 14))
        self.className = Label(self.master, text="Class :",font=("Comic Sans", 14))
        self.netValue_value = Label(self.master, text="",font=("Comic Sans", 14))
        self.className_value = Label(self.master, text="",font=("Comic Sans", 14))




        # positions
        self.classes.place(relx=1, x=-530, y= 100, anchor=NE)
        self.features.place(relx=1, x=-685, y=100, anchor=NE)
        self.epochs.place(relx=1, x=-325, y=100, anchor=NE)
        self.learningrate.place(relx=1, x=-175, y=100, anchor=NE)
        self.label.place(relx=1, x=-250, y=20, anchor=NE)
        self.biaslabel.place(relx=1, x=-385, y=155, anchor=NE)
        self.stats.place(relx=1, x=-655, y=290, anchor=NE)
        self.Accuracy.place(relx=1, x=-650, y=335, anchor=NE)
        self.Accuracy_value.place(relx=1, x=-600, y=335, anchor=NE)
        self.Tests.place(relx=1, x=-300, y=290, anchor=NE)
        self.X1value.place(relx=1, x=-400, y=325, anchor=NE)
        self.X2value.place(relx=1, x=-400, y=375, anchor=NE)
        self.netValue.place(relx=1, x=-400, y=430, anchor=NE)
        self.className.place(relx=1, x=-400, y=480, anchor=NE)
        self.netValue_value.place(relx=1, x=-255, y=430, anchor=NE)
        self.className_value.place(relx=1, x=-255, y=480, anchor=NE)





    def createCombobox(self):

        self.feature1_value = StringVar()
        self.feature2_value = StringVar()
        self.class1_value = StringVar()
        self.class2_value = StringVar()
        self.yesValue = IntVar()
        self.noValue = IntVar()


        self.feature1ComboBox = ttk.Combobox(self.master, textvariable= self.feature1_value, state='readonly')
        self.feature2ComboBox = ttk.Combobox(self.master, textvariable= self.feature2_value, state='readonly')
        self.class1Combobox = ttk.Combobox(self.master, textvariable= self.class1_value, state='readonly')
        self.class2Combobox = ttk.Combobox(self.master, textvariable= self.class2_value, state='readonly')
        self.learningrate_textbox = Entry(self.master)
        self.epochs_textbox = Entry(self.master)
        self.X1_textbox = Entry(self.master)
        self.X2_textbox = Entry(self.master)
        self.yesValue_CB = Checkbutton(self.master, text="Yes", variable=self.yesValue)
        self.noValue_CB = Checkbutton(self.master, text="No", variable=self.noValue)

        self.feature1ComboBox['values'] = ('X1', 'X2', 'X3', 'X4')
        self.feature2ComboBox['values'] = ('X1', 'X2', 'X3', 'X4')
        self.class1Combobox['values'] = ('Iris-setosa', 'Iris-versicolor', 'virginica')
        self.class2Combobox['values'] = ('Iris-setosa', 'Iris-versicolor', 'virginica')

        self.feature1ComboBox.current(0)
        self.feature2ComboBox.current(1)
        self.class1Combobox.current(0)
        self.class2Combobox.current(1)


        self.feature1ComboBox.place(relx=1, x=-650, y=130, anchor=NE)
        self.feature2ComboBox.place(relx=1, x=-650, y=180, anchor=NE)
        self.class1Combobox.place(relx=1, x=-490, y=130, anchor=NE)
        self.class2Combobox.place(relx=1, x=-490, y=180, anchor=NE)
        self.epochs_textbox.place(relx=1, x=-330, y=130, anchor=NE)
        self.learningrate_textbox.place(relx=1, x=-160, y=130, anchor=NE)
        self.X1_textbox.place(relx=1, x=-270, y=330, anchor=NE)
        self.X2_textbox.place(relx=1, x=-270, y=380, anchor=NE)
        self.yesValue_CB.place(relx=1, x=-425, y=180, anchor=NE)
        self.noValue_CB.place(relx=1, x=-355, y=180, anchor=NE)


    def start_trainning(self):
        f1_index = self.feature1ComboBox.current()
        f2_index = self.feature2ComboBox.current()
        class1Name = self.class1Combobox.get()
        Class2Name = self.class2Combobox.get()
        indexes = []
        indexes.append(f1_index)
        indexes.append(f2_index)
        classes = []
        classes.append(class1Name)
        classes.append(Class2Name)
        task1.Task1NeuralNetwork.fit(classes,self.yesValue.get(),self.learningrate_textbox.get(),int(self.epochs_textbox.get()),f1_index,f2_index)
        accuracy = task1.Task1NeuralNetwork.accuracy
        accuracy = accuracy * 100
        accuracy = str(accuracy) + "%"
        self.Accuracy_value.config(text=accuracy)

    def draw_dataset(self):

        f1_index = self.feature1ComboBox.current()
        f2_index = self.feature2ComboBox.current()
        c1_index = self.class1Combobox.current()
        c2_index = self.class2Combobox.current()
        c1_color , c1_data = self.getClassColor(c1_index)
        c2_color , c2_data = self.getClassColor(c2_index)
        self.selectFeature(c1_data,c2_data,f1_index,f2_index)

        if f1_index == f2_index or c1_index == c2_index:
           messagebox.showerror('Are you good today ?','Cant Draw on The same features or classes ?!')
        else:
            plt.xticks(np.arange(0, 10, 0.4))
            plt.yticks(np.arange(0, 10, 0.4))
            plt.xlim(0.0, 7.5)
            plt.ylim(0.0, 7.5)
            plt.plot(self.c1_f1_data, self.c1_f2_data, c1_color)
            plt.plot(self.c2_f1_data, self.c2_f2_data, c2_color)
            plt.show()

    def getClassColor(self, index):

        if index == 0:
            return 'ro', self.class1trainningset
        elif index == 1:
            return 'go', self.class2trainningset
        else:
            return 'bo', self.class3trainningset

    def selectFeature(self , c1_data, c2_data, f1_index, f2_index):
        self.c1_f1_data = []
        self.c2_f1_data = []
        self.c1_f2_data = []
        self.c2_f2_data = []
        for i in range (30):
            self.c1_f1_data.append(c1_data[i][f1_index])
            self.c2_f1_data.append(c2_data[i][f1_index])
            self.c1_f2_data.append(c1_data[i][f2_index])
            self.c2_f2_data.append(c2_data[i][f2_index])

    def test_features(self):
        fe = []
        first = self.X1_textbox.get()
        second = self.X2_textbox.get()
        fe.append(first)
        fe.append(second)
        predictionval, netval = task1.Task1NeuralNetwork.predict(fe)
        self.netValue_value.config(text= str(netval))
        if predictionval == -1:
            self.className_value.config(text= self.class1Combobox.get())
        else:
            self.className_value.config(text= self.class2Combobox.get())













root = Tk()
root.title("Neural Network Task 1")
root.geometry('800x600')
my_gui = Task1Gui(root)
root.mainloop()