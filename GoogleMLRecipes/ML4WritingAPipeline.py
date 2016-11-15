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

#classifier specific code
#decision tree
# from sklearn import tree
# #create classifier - DecisionTreeClassifier
# my_classifier = tree.DecisionTreeClassifier()

#k-nearest neighbour tree
from sklearn.neighbors import KNeighborsClassifier
#create classifier - DecisionTreeClassifier
my_classifier = KNeighborsClassifier()

# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# my_classifier = RandomForestClassifier()
# #my_classifier = AdaBoostClassifier()

#train or find patterns or build rule on training data
my_classifier = my_classifier.fit(X_train,y_train)
#predict the test data
prediction = my_classifier.predict(X_test)
#print the prediction
print(prediction)

#calculate the accuracy of the prediction
from sklearn.metrics import accuracy_score
#the utility function compares the true label we generated with the predicted ones
<<<<<<< HEAD
print (accuracy_score(y_test, prediction))
=======
print accuracy_score(y_test, prediction)
>>>>>>> b91ac460ff05a23aba380cedf7a47f2feff8f1fe
