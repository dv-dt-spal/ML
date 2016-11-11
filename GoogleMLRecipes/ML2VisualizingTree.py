import numpy as np
#Iris is a sample datasets provides by scikit learn
from sklearn.datasets import load_iris
from sklearn import tree

#IMPORT DATASET
#Load the iris data set
iris = load_iris()
#index of variables to be deleted for testing
test_idx =  [0,50,100]
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])
#training data - created from the original data by removing 3 entries
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis = 0)

#testing data created from the deleted 3 index from original iris dataset
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
print("Three Flowes " ,iris.target_names)
print(test_target)
print("4 descriptions for three flowers ", iris.feature_names)
print(test_data)
#IMPORT DATASET

#TRAIN CLASSIFIER
#Create the empty classifier
clf = tree.DecisionTreeClassifier()
#Train on the training data which is the one removed with 3 entries
clf = clf.fit(train_data,train_target)
#TRAIN CLASSIFIER

#PREDICT THE FLOWER BASED ON THE TEST DATASET
print("Predicting the test data, output are labels ",clf.predict(test_data))
#PREDICT THE FLOWER BASED ON THE TEST DATASET


#VISUALIZE THE DECISION TREE
import matplotlib.pyplot as plt
#VISUALIZE THE DECISION TREE
