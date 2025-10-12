import numpy as np

# 初始化神经网络的权重和偏置
def init_network():
    network = {} # 创建一个空字典
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 第一层权重, 前面有2个神经元，后面有3个神经元，所以是2x3的矩阵
    network['b1'] = np.array([0.1, 0.2, 0.3]) # 第一层偏置, 后面有3个神经元，到每一个神经元都有一个偏置，所以是3x1的矩阵
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 第二层权重, 前面有3个神经元，后面有2个神经元，所以是3x2的矩阵
    network['b2'] = np.array([0.1, 0.2]) # 第二层偏置, 后面有2个神经元，到每一个神经元都有一个偏置，所以是2x1的矩阵
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 第三层权重, 前面有2个神经元，后面有2个神经元，所以是2x2的矩阵
    network['b3'] = np.array([0.1, 0.2]) # 第三层偏置, 后面有2个神经元，到每一个神经元都有一个偏置，所以是2x1的矩阵
    return network


def forward(network, x):
    # network 是一个字典，包含了神经网络的所有权重和偏置
    # x 是输入数据，类型是 numpy 数组
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1 # 输入 x 数组 和 第一层权重 W1 做点积，再加上 第一层偏置 b1
    z1 = sigmoid(a1) # 第一层的输出，经过激活函数 sigmoid
    a2 = np.dot(z1, W2) + b2 # 第一层的输出 z1 和 第二层权重 W2 做点积，再加上 第二层偏置 b2
    z2 = sigmoid(a2) # 第二层的输出，经过激活函数 sigmoid
    a3 = np.dot(z2, W3) + b3 # 第二层的
    y = identity_function(a3) # 第三层的输出，经过恒等函数
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

network = init_network()
x = np.array([1.0, 0.5]) # 输入数据，类型是 numpy 数组
y = forward(network, x) # 计算神经网络的输出
print(y)
