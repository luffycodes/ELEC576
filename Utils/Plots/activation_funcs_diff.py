# Credit: https://medium.com/@curiousily/tensorflow-for-hackers-part-iv-neural-network-from-scratch-1a4f504dfa8
# Credit: http://jiffyclub.github.io/numpy/reference/generated/numpy.vectorize.html

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

from Utils.maths import *

x = np.linspace(-10., 10., num=100)
sig = sigmoid(x)
sig_prime = diff_sigmoid(x)

plt.plot(x, sig, label="sigmoid")
plt.plot(x, sig_prime, label="sigmoid prime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size' : 16})
plt.show()
