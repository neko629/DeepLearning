import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

x = np.arange(0, 6, 0.1)
y = np.sin(x)
z = np.cos(x)

# 绘制
plt.plot(x, y, label='sin(x)')
plt.plot(x, z, linestyle = "--", label='cos(x)')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.show()
