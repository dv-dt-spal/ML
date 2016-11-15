import random
from scipy.spatial import distance

#This function will measure the distance between two features regardless of
#dimension. So for any k nearest neighbour we will be able to find the number of
#closest neighbours and vote to classify the point
def euc(a,b):
    return distance.euclidean(a,b)


#KNN classifier
class MyKNN():
    #Function to find pattern and form the rule box
    #Takes Features and labels as input
    def fit(self,X_train,y_train):
        #Store the data in the class
        self.X_train = X_train
        self.y_train = y_train

    #Loop over all the training point and keep track of the closest point
    def closest(self, row):
    #Variable to have the best distance from the test dataset
        best_dist = euc(row, self.X_train[0])
    #variable to store the index
        best_index = 0;
        for idx in range(1, len(self.X_train)):
            curr_dist = euc(row, self.X_train[idx])
            if(best_dist > curr_dist):
                best_dist = curr_dist
                best_index = idx
        return y_train[best_index]

    #Function to return the output/label with the feature data provided
    def predict(self,X_test):
        #List of predictions as X_test is a list of list
        predictions = []
        #1NN from the training labels calculate the distance from all training
        #data and label the one with the closet point
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        #RANDOM IMPLEMENTATION VERY BAD ACCURACY
        #Random selection from the training labels
        #RANDOM IMPLEMENTATION VERY BAD ACCURACY
        # for row in X_test:
        #     label = random.choice(self.y_train)
        #     predictions.append(label)
        #RANDOM IMPLEMENTATION VERY BAD ACCURACY
        return predictions

#import dataset
from sklearn import datasets
iris = datasets.load_iris()

#f(x) = y - classifier as function
#Features - x
X = iris.data
#Labels - y
y = iris.target

#partition the dataset to train and test by importing a handy utility
from sklearn.cross_validation import train_test_split
# half the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .5)

#k-nearest neighbour tree
my_classifier = MyKNN()

# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# my_classifier = RandomForestClassifier()
# #my_classifier = AdaBoostClassifier()

#train or find patterns or build rule on training data
my_classifier.fit(X_train,y_train)
#predict the test data
prediction = my_classifier.predict(X_test)
#print the prediction
print(prediction)

#calculate the accuracy of the prediction
from sklearn.metrics import accuracy_score
#the utility function compares true label we generated with the predicted ones
print (accuracy_score(y_test, prediction))
