from sklearn import metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import learn

#partition the dataset to train and test by importing a handy utility
from sklearn.cross_validation import train_test_split

def main(unused_argv):
    #lead dataset
    iris = learn.datasets.load_dataset('iris')
    x_train,x_test,y_train,y_test = train_test_split(
        iris.data, iris.target, test_size=0.2)

    #Build a 3 layer DNN with 10 20 10 units respectively
    classifier = learn.DNNClassifier(hidden_units=[10,20,10],n_classes=3)

    #Fit and predict
    classifier.fit(x_train,y_train,steps=1)
    score = metrics.accuracy_score(y_test,classifier.predict(x_test))
    print('Accuracy: {0:f}'.format(score))
