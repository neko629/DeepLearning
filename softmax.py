import numpy as np


def softmax(a):
    c = np.max(a) # 防止溢出
    exp_a = np.exp(a - c) # 每个元素都减去最大值
    sum_exp_a = np.sum(exp_a) # 所有元素的和
    y = exp_a / sum_exp_a # 每个元素都除以总和
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a) # 输出结果是0~1之间的数，且和为1
print(y)
print(np.sum(y))
