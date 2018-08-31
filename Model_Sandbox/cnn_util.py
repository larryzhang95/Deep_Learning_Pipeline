import numpy as np
#import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten, Permute, Reshape
from keras.layers import Conv2D, MaxPooling2D
import sys
import sklearn
from sklearn.model_selection import train_test_split

class Data_Preprocessor(object):
    '''
    Data Preprocessor is a class that is enabled to pre-process any sets of Data
    '''
    def __init__(self,train_data=None, test_data=None , val_data=None):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        if label == None:
            print("Please provide a set of labels to the object")
            sys.exit(1)
        self.train_data,self.test_data,self.val_data = self.train_test_validate()

    def train_test_validate(self):
        '''
        Typically what is expected is 1 of 3 cases:

        Case 1: Data provided all in 1 set of data (ie: pickle source.; csv; etc.)
            - Data needs to be split into train_test_validation set

        Case 2: Data provided in 2 sets of data
            - Test Data can Stay as is, however, create a 10% validation set on the
            training set to verify and validate the model

        Case 3: Data provided in 3 sets. This means there is a train, test, and val set

        Data specified should already be unpacked.
        '''
        if self.train_data == None:
            print("Please provide a Training Data set.")
            sys.exit(1)
        else:
            if self.test_data == None and self.val_data == None:
                print("Case 1: Only 1 Source of Data")
                print(r"Data Split into 60% Training : 20 % Test : 20 % Validation")
                train,test = train_test_split(self.train_data,test_size=0.4)
                test,validate = train_test_split(test,test_size=0.5)
                return train, test, validate
            elif self.test_data != None and self.val_data == None:
                print("Case 2: Train and Test Source Exist")
                train,validate = train_test_split(self.train_data,test_size=0.1)
                return train, self.test_data, validate
            elif self.test_data != None and self.val_data != None:
                print("Case 3: Train, Test ,Validation all Exist")
                return self.train_data, self.test_data, self.val_data

    def data_label_split(self,data):
        '''
        The data_label_split is a command that should enable the splitting of
        sets of train,test, and validation data. Data is provided one set at a time.
        Cases:
        '''

    def reformat_data(self,data):
        print("YEET")

class CNN_Model(object):
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.data_dimensions = (32,32,10)
        self.model = Sequential()

    def build_model(self):
        model = self.model
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,10)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model

        
    def fit_model(self,X_test,y_test):
        '''
        Fits model on top of data
        '''
        model = self.model
        model.fit(self.X_train, self.y_train, batch_size=32, epochs=20)
        self.score = model.evaluate(x_test, y_test, batch_size=32)
        return
