import numpy as np


# def identity_function(x):
#     return x


# def step_function(x):
#     return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    x = x - np.max(x)  # 溢出对策
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:  # 确保y是二维的
        y = y.reshape(1, y.size)
        
    if t.ndim == 1:  # 如果t是一维的，它已经是索引形式
        pass
    elif t.ndim == 2 and t.size == y.size:  # 如果t是二维的并且大小匹配y，转换为索引
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    # 计算交叉熵损失
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
