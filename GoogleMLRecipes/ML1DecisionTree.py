from sklearn import tree
#Generic as it can be changed for various kind of training data
#This will be the input to the classfier [weight,bumpy||smooth]
features = [[140,1], [130,1],[150,0],[170,0]]
#This will be the output of the classfier 0 apple 1 orange
labels = [0,0,1,1]

# #This will be the input to the classfier [Horsepower,seat]
# features = [[300,2], [450,2],[200,8],[150,9]]
# #This will be the output of the classfier sports car , minivan
# labels = ["sports car","sports car","minivan","minivan"]

#Create the empty classfier or empty box of rules
clf = tree.DecisionTreeClassifier()

#In Scikit the training algorithm is included in the classifier object
#it is called fit. It is synonym to find patterns in data
#We are creating a learning algorithm here which will form the box of rules
#clf will be the trained classifer
clf = clf.fit(features,labels)

#To test the classifer with some random data
print(clf.predict([[500,2]]))
