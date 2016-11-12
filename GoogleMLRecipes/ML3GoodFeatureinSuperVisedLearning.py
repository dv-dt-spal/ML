import numpy as np
import matplotlib.pyplot as plt

#number of dogs
greyhounds = 500
labs = 500

#height of the dogs = assumed height + random normal distribution for variation
gray_height = 28 + 4 * np.random.randn(greyhounds)
lab_height  = 24 + 4 * np.random.randn(labs)

#plot the histogram based on the array of numbers representing the dog height
plt.hist([gray_height,lab_height],stacked = True,color = ['r','b'])
plt.show()
