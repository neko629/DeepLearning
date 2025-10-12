import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)
plt.plot(x, y)
plt.plot(x, y2, linestyle='--')
plt.plot(x, y3, linestyle="--")
#plt.ylim(-1.5, 1.5)
plt.show()