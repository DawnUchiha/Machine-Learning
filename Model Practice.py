import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from sklearn import preprocessing
from matplotlib import style
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import os
from itertools import chain
import random


dt = True
while dt:
    dta = input("What is the name of the dataset file?(with the file.type): ")
    if os.path.exists(f'{dta}'):
        data =  dta
        sep = input("What is the separator?(eg: , or ;): ")
        dt = False
        
    else:
        print("it doesn't exist try again.")


class Label():

    def __init__(self) -> None:
        self.x 
        self.y
        self.alllabels
        

    def labeling(self, clsn):
            
        data1 = pd.read_csv(data)
        
        if clsn == 1:
            print("I dont think that would work then")
        

        elif clsn == 2:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Predicted label: ")
            
            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            lp = le.fit_transform(list(data1[i2]))

            self.x = list(zip(l1))
            self.y = list(lp)
            self.alllabels = [i1,i2]


        elif clsn == 3:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            lp = le.fit_transform(list(data1[i3]))
            
            self.x = list(zip(l1, l2))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3]


        elif clsn == 4:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            lp = le.fit_transform(list(data1[i4]))
            
            self.x = list(zip(l1, l2, l3))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4]


        elif clsn == 5:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            lp = le.fit_transform(list(data1[i5]))

            self.x = list(zip(l1,l2,l3,l4))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5]


        elif clsn == 6:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Fifth label: ")
            i6 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            l5 = le.fit_transform(list(data1[i5]))
            lp = le.fit_transform(list(data1[i6]))

            self.x = list(zip(l1, l2, l3, l4, l5))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5,i6]


        elif clsn == 7:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Fifth label: ")
            i6 = input("Name of the Sixth label: ")
            i7 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            l5 = le.fit_transform(list(data1[i5]))
            l6 = le.fit_transform(list(data1[i6]))
            lp = le.fit_transform(list(data1[i7]))

            self.x = list(zip(l1, l2, l3, l4, l5, l6))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5,i6,i7]


        elif clsn == 8:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Fifth label: ")
            i6 = input("Name of the Sixth label: ")
            i7 = input("Name of the Seventh label: ")
            i8 = input("Name of the Predicted label: ")
            
            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            l5 = le.fit_transform(list(data1[i5]))
            l6 = le.fit_transform(list(data1[i6]))
            l7 = le.fit_transform(list(data1[i7]))
            lp = le.fit_transform(list(data1[i8]))

            self.x = list(zip(l1, l2, l3, l4, l5, l6, l7))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5,i6,i7,i8]
  

        elif clsn == 9:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Fifth label: ")
            i6 = input("Name of the Sixth label: ")
            i7 = input("Name of the Seventh label: ")
            i8 = input("Name of the Eighth label: ")
            i9 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            l5 = le.fit_transform(list(data1[i5]))
            l6 = le.fit_transform(list(data1[i6]))
            l7 = le.fit_transform(list(data1[i7]))
            l8 = le.fit_transform(list(data1[i8]))
            lp = le.fit_transform(list(data1[i9]))

            self.x = list(zip(l1, l2, l3, l4, l5, l6, l7, l8))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5,i6,i7,i8,i9]

        elif clsn == 10:
            i1 = input("Name of the First label: ")
            i2 = input("Name of the Second label: ")
            i3 = input("Name of the Third label: ")
            i4 = input("Name of the Fourth label: ")
            i5 = input("Name of the Fifth label: ")
            i6 = input("Name of the Sixth label: ")
            i7 = input("Name of the Seventh label: ")
            i8 = input("Name of the Eighth label: ")
            i9 = input("Name of the Ninth label: ")
            i10 = input("Name of the Predicted label: ")

            le = preprocessing.LabelEncoder()
            l1 = le.fit_transform(list(data1[i1]))
            l2 = le.fit_transform(list(data1[i2]))
            l3 = le.fit_transform(list(data1[i3]))
            l4 = le.fit_transform(list(data1[i4]))
            l5 = le.fit_transform(list(data1[i5]))
            l6 = le.fit_transform(list(data1[i6]))
            l7 = le.fit_transform(list(data1[i7]))
            l8 = le.fit_transform(list(data1[i8]))
            l9 = le.fit_transform(list(data1[i9]))
            lp = le.fit_transform(list(data1[i10]))

            self.x = list(zip(l1, l2, l3, l4, l5, l6, l7, l8, l9))
            self.y = list(lp)
            self.alllabels = [i1,i2,i3,i4,i5,i6,i7,i8,i9]

        elif clsn == 0:
            i1 = input("What is the Target Label")

            lp = le.fit_transform(list(data1[i1]))
            self.y = list(lp)

        elif clsn > 10:
            print("That is more than 10.")
        else:
            print("I don't think you put the right thing.")

def knn_model(data, n):
    

    data = pd.read_csv(data, sep=sep)
    
     
    classes = input("How many attribute including the target are there, note for this no more than 10: ")
    classes = int(classes)
    Label.labeling(Label, classes)
    x, y =  Label().x, Label().y
    
    
    rl = True
    while rl:
       
        it = input("Is this model already trained?(Yes = y, No = n): ") 
        if it == 'n':
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
            model = KNeighborsClassifier(n_neighbors=n)

            model.fit(x_train, y_train)
            acc = model.score(x_test, y_test)
            print(acc)

            predicted = model.predict(x_test)
            
            for x in range(len(predicted)):
                print("Data: ", x_test[x], "Predicted Data: ", predicted[x],  "Actual: ", y_test[x])
            
            tl = 1000
            best = 0
            rt = input(f"How many times do you want it to train?(min = 1, max = {tl})")
            if rt.isdigit():
                rt = int(rt)

                if rt < 1:
                    print('That is less 1')
                elif rt > tl:
                    print(f'That is more than {tl}')
            
            pn = input("What do you want to name the file?: ")
            for _ in range(rt):

                

                knn = model

                knn.fit(x_train, y_train)
                accr = knn.score(x_test, y_test)
                print(accr)

                if accr > best:
                    best = accr
                    
                    with open(f"{pn}.pickle", "wb") as f:
                        pickle.dump(knn, f)

            
        elif it == 'y':
            
            pt = True
            while pt:
                tpn = input('What is the name of the trained pickle file?(Without the .pickle): ')
                if os.path.exists(f'{tpn}.pickle'):
                    ttpn = tpn + '.pickle'
                    pt = False
                else:
                    print("it doesn't exist try again.")
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
            pickle_in = open(ttpn, "rb")

            knn =pickle.load(pickle_in)
            knn.fit(x_train, y_train)
            acc = knn.score(x_test, y_test)
            print(acc)

            

            predictions = knn.predict(x_test)

            for x in range(len(predictions)):
                print(predictions[x], x_test[x], y_test[x])
            if input("Do you want to see it plotted?(y or n): ") == 'y':
                """This is what puts the data on a graph"""
                p = input("What of the attributes are you comparing to the predicted attribute?(Note: Has to be on of the previously selected): ")#This is the value we are comparing the Final to
                yn = input("What is Predicted Attribute?: ")
                if p in Label().alllabels:
                    if yn in Label().alllabels:
                        namep = input("Name the Comparing Attribute for show: ")
                        namey = input("Name the Predicted Attribute for show: ")

                        style.use("ggplot")
                        pyplot.scatter(data[p], data[yn])
                        pyplot.xlabel(namep)
                        pyplot.ylabel(namey)
                        pyplot.show()

        while True:
            p = input("Do you want to change the num of Neighbors?(y or n)(!!!Don't try change it need to be fixed sorry): ")
            if p == 'y':
                n = input("What is the num you want it to be?: ")
                n = int(n)
                break
            elif p == 'n':
                rl = False
                break
            else:
                print("Sorry what was that?")
    
def linearReg(data):
    data = pd.read_csv(data, sep=sep)

    classes = input("How many attribute including the target are there, note for this no more than 10: ")
    classes = int(classes)
    Label.labeling(Label, classes)
    x, y =  Label().x, Label().y

    train = input("Do you want open or train a model?(o or t): ")
    if train == 't':
        best = 0
        while True:
            maxt =1000
            mint = 1
            r = input('How many times do you want to train this model?(Min 1, Max 1000): ')
            if r.isdigit():
                r = int(r)
                if r < 1:
                    print("Too small")
                if r > maxt:
                    print("Too big")
                else:
                    name = input('What do you want to name the pickle?(Without the .pickle): ')
                    break
            else:
                print('That is not a number.')

        for _ in range(r):

            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

            linear = linear_model.LinearRegression()

            linear.fit(x_train, y_train)
            acc = linear.score(x_test, y_test)
            print(acc)

            if acc > best:
                best = acc
                
                with open(f"{name}.pickle", "wb") as f:
                    pickle.dump(linear, f)

    elif train == 'o':
        name = input("What is the name of the .pickle file(Without the .pickle)")
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
        pickle_in = open(f"{name}.pickle", "rb")

        linear =pickle.load(pickle_in)

        predictions = linear.predict(x_test)

        for x in range(len(predictions)):
            print(predictions[x], x_test[x], y_test[x])

    if input("Do you want to see it plotted?(y or n): ") == 'y':
        """This is what puts the data on a graph"""
        p = input("What of the attributes are you comparing to the predicted attribute?(Note: Has to be on of the previously selected): ")#This is the value we are comparing the Final to
        yn = input("What is Predicted Attribute?: ")
        if p in Label().alllabels:
            if yn in Label().alllabels:
                namep = input("Name the Comparing Attribute for show: ")
                namey = input("Name the Predicted Attribute for show: ")

                style.use("ggplot")
                pyplot.scatter(data[p], data[yn])
                pyplot.xlabel(namep)
                pyplot.ylabel(namey)
                pyplot.show()
    
def svm_model(data):


    data = pd.read_csv(data, sep=sep)
    
     
    classes = input("How many attribute including the target are there, note for this no more than 10: ")
    classes = int(classes)
    Label.labeling(Label, classes)
    x, y =  Label().x, Label().y
    
    
    rl = True
    while rl:
       
        it = input("Is this model already trained?(Yes = y, No = n): ") 
        if it == 'n':
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
            k = input("What do you want the kernal to be?(linear, poly, rbf, sigmoid, precomputed): ")
            while True:
                c = input("what do you want the Regularization parameter to be(Must be Positive num): ")
                if c.isdigit():
                    c = int(c)
                    if c >= 1:

                        break
                    else:
                        print("That needs to be a positve")
                else: 
                    print("That is not a number")
            model = svm.SVC(kernel=k, C=c)

            model.fit(x_train, y_train)
            acc = model.score(x_test, y_test)
            print(acc)

            predicted = model.predict(x_test)
            
            
            
            tl = 1000
            best = 0
            rt = input(f"How many times do you want it to train?(min = 1, max = {tl})")
            if rt.isdigit():
                rt = int(rt)

                if rt < 1:
                    print('That is less 1')
                elif rt > tl:
                    print(f'That is more than {tl}')
            
            pn = input("What do you want to name the file?: ")
            for _ in range(rt):

                

                sv = model

                sv.fit(x_train, y_train)
                accr = sv.score(x_test, y_test)
                print(accr)

                if accr > best:
                    best = accr
                    
                    with open(f"{pn}.pickle", "wb") as f:
                        pickle.dump(sv, f)
            for x1 in range(len(predicted)):
                print("Data: ", x_test[x1], "Predicted Data: ", predicted[x1],  "Actual: ", y_test[x1])

            
        elif it == 'y':
            
            pt = True
            while pt:
                tpn = input('What is the name of the trained pickle file?(Without the .pickle): ')
                if os.path.exists(f'{tpn}.pickle'):
                    ttpn = tpn + '.pickle'
                    pt = False
                else:
                    print("it doesn't exist try again.")
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
            pickle_in = open(ttpn, "rb")

            sv =pickle.load(pickle_in)
            sv.fit(x_train, y_train)
            acc = sv.score(x_test, y_test)
            print(acc)

            

            predictions = sv.predict(x_test)

            for x2 in range(len(predictions)):
                print(predictions[x2], x_test[x2], y_test[x2])
            if input("Do you want to see it plotted?(y or n): ") == 'y':
                """This is what puts the data on a graph"""
                p = input("What of the attributes are you comparing to the predicted attribute?(Note: Has to be on of the previously selected): ")#This is the value we are comparing the Final to
                yn = input("What is Predicted Attribute?: ")
                if p in Label().alllabels:
                    if yn in Label().alllabels:
                        namep = input("Name the Comparing Attribute for show: ")
                        namey = input("Name the Predicted Attribute for show: ")

                        style.use("ggplot")
                        pyplot.scatter(data[p], data[yn])
                        pyplot.xlabel(namep)
                        pyplot.ylabel(namey)
                        pyplot.show()
        else:
            break
    
def kmeans_model(data):
    print("Not working Sorry")

while True:
    wm = input("Which model do you want to use for your dataset?(Note this version's models have a limit of 10 Attributtes)\n(| Knn(knn) | LinearR(lr) | SVM(svm) | Kmeans(km) |)")

    if wm == 'knn':
        n = int(input('How many Neighbors: '))
        knn_model(data, n)
    if wm == 'lr':
        linearReg(data)
    if wm == 'svm':
        svm_model(data)
    if wm == 'km':
        kmeans_model(data)
    if input("Do you want to quit?(q): ") == 'q':
        break
    else:
        pass




