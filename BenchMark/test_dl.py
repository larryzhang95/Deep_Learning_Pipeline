import numpy as np
import sys
from typing import List
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import matplotlib.pyplot as plt
'''
TODOs:

Goal of test_dl:

Determine and establish, over given set of data, whether or not Deep Learning is a viable option.
This means that it proves to output and generate better results, as the data scales up

==============================================
Methodology:
Given any arbitrary set of data (50k+ datapoints.), we will split the dataset into thousand(s)
50/50 Train Set vs. Test Set.

We will run the data through a set of algorithm(s), and track the performance of the algorithms across time.
- If it proves that the more data accumulated, the better performance of Deep Learning Model(s), and eventually
crosses over past Linear and Non-Linear Model(s)

Models used:
- Logistic Regression
- SVM (rbf filter)
- LeNet? - or similar ConvNet
'''

class Test_DL(object):
    def __init__(self,data,labels=None,targets=None):
        '''
        In the event that there is a set of labels. Typically it will be a list
        When there's targets, then we're dealing with a dataframe.
        '''
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data = data
        self.size_list = None
        if labels != None:
            self.labels = labels
            if targets != None:
                print("If the object has labels, there should be no targets")
                print("Exiting with Status 1")
                sys.exit(1)
            else:
                self.data = np.hstack((self.data,self.labels))
                self.size_list = np.arange(2500,self.data.shape[0],2500)
        if labels == None:
            if targets == None:
                print("Object must have labels or targets to develop labels from")
                print("Exiting with Status 1")
                sys.exit(1)
            else:
                self.targets = targets
                self.size_list = np.arange(2500,len(self.data),2500)
        self.result_dict = []

    def seperate_data(self, size):
        '''
        Splits data into sets of defined size, given the sample methodology.
        Called each time run on data.

        Size at most can be size(data)/2

        Data Fed in is either Array or Data Frame.
        '''
        data_size = len(self.data)
        if size > data_size/2:
            print("Size of portion is greater than size of data.")
            print("Please check on the size parameter passed to method")
            print("Exiting with Status 1")
        else:
            if isinstance(self.data,pd.DataFrame):
                sample_set = self.data.sample(n=size * 2)
                train_set, test_set = np.split(sample_set,[int(.5*len(df))])
            elif isinstance(self.data,List):
                sample_set = self.data[np.random.randint(self.data.shape[0], size=size*2), :]
                train_test = sample_set[:size]
                test_set = sample_set[size:]
            else:
                print("Data may not be in correct format. Please try again")
                print("Exiting with Status 1")
                sys.exit(1)
        return train_set, test_set

    def split_data(self,data_set):
        if isinstance(data_set,pd.DataFrame):
            y = data_set[self.labels]
            X = data_set.drop(self.labels)

        if isinstance(self.data,List):
            y = data_set[:,-1]
            X = data_set[:,[:-1]]
        return X,y

    def setup_data(self,size):
        train_set, test_set = self.seperate(size)
        self.X_train, self.y_train = split_data(train_set)
        self.X_test, self.y_test = split_data(test_set)

    def logistic_regression(self):
        tuned_parameters = [{'penalty':['l1','l2'],'C':np.logspace(0,4,10)}]
        clf = GridSearchCV(LogisticRegression,tuned_parameters)
        clf.fit(self.X_train,self.y_train)
        y_test_pred = clf.predict(self.X_test)
        test_accuracy = np.mean(y_test_pred == self.y_test)
        return test_accuracy

    def random_forest(self):
        tuned_parameters = [{'n_estimators':[10,100,250,500,1000,2000],'max_features':['auto','sqrt'],'max_depth':[2,5,10,20],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4]}]
        clf = GridSearchCV(RandomForestClassifier(),tuned_parameters)
        clf.fit(self.X_train,self.y_train)
        y_test_pred = clf.predict(self.X_test)
        test_accuracy = np.mean(y_test_pred == self.y_test)
        return test_accuracy

    def CNN(self):

    def run_models(self):
        '''
        Runs suite of model(s) by size of training and test sets.
        Used to validate the benefits of linear vs. nonlinear vs. NN models

        Dictionary returns: {'set_size':<size>,'model':<model_name>,'acc':<accuracy>}
        '''
        for size in self.size_list:
            self.setup_data(size)
            lr_acc = self.logistic_regression()
            lr_dict = {'set_size':size, 'model':'Logistic_Regression', 'accuracy':lr_acc}
            rf_acc = self.random_forest()
            rf_dict = {'set_size':size, 'model':'Random_Forest', 'accuracy':rf_acc}
            #cnn_acc = self.CNN()
            #cnn_dict = {'set_size':size, 'model':'Conv_Neural_Net', 'accuracy':cnn_acc}
            self.result_dict.append(lr_dict)
            self.result_dict.append(rf_dict)
            #self.result_dict.append(cnn_dict)

    def plot_results(self,lr_list,rf_list,nn_list):
        '''
        Plots the result(s) in three different graphs for visualization of performance
        '''
        size_list = []
        acc_list = []
        for d in lr_list:
            size_list.append(d['size'])
            acc_list.append(d['accuracy'])
        plt.plot(size_list,acc_list, 'ro')

        size_list = []
        acc_list = []
        for d in rf_list:
            size_list.append(d['size'])
            acc_list.append(d['accuracy'])
        plt.plot(size_list,acc_list, 'b')

        # size_list = []
        # acc_list = []
        # for d in nn_list:
        #     size_list.append(d['size'])
        #     acc_list.append(d['accuracy'])
        #plt.plot(size_list,acc_list, 'y')
        plt.show()

    def evaluate_results(self):
        '''
        Evaluates the result(s) given by the run_models command. Reports best performing
        Model over time @ each step.

        Calls plot_results to graph data.

        Checks if there is anything in self.result_dict. If nothing in it, means run_models
        hasn't been called yet.

        At the end, empty the result_dict
        '''
        model_list = ['Logistic_Regression','Random_Forest','Conv_Neural_Net']
        lr_list = []
        rf_list = []
        nn_list = []
        best_perf_list = []
        if len(self.result_dict) == 0:
            print("The command run_models has not been instantiated yet.")
            print("Must call run_models prior to evaluating results")
            sys.exit(1)
        else:
            for dict in self.result_dict:
                if dict['model'] = model_list[0]:
                    lr_list.append(dict)
                if dict['model'] = model_list[0]:
                    rf_list.append(dict)
                if dict['model'] = model_list[0]:
                    nn_list.append(dict)
            for i in range(len(lr_list)):
                lr_acc = lr_list[0]['accuracy']
                rf_acc = rf_list[0]['accuracy']
                #nn_acc = nn_list[0]['accuracy']
                if lr_acc > rf_acc and lr_acc>nn_acc:
                    best_perf_list.append('Logistic_Regression')
                elif rf_acc > lr_acc and rf_acc > nn_acc:
                    best_perf_list.append('Random_Forest')
                else:
                    best_perf_list.append('Neural_Network')
            print(best_perf_list)
            self.plot_results(lr_list,rf_list,nn_list)
            self.result_dict = []
