import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class FlexibleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 pool_params = {'pool_size': 2, 'stride': 2},
                 conv_params=[
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                     {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}
                 ],weight_init_std=0.01,
                 hidden_size=50, output_size=10):
        
        self.params = {}
        self.conv_layer = []
        affine_start = len(conv_params) + 1
        pre_channel_num = input_dim[0]

        for idx, conv_param in enumerate(conv_params):
            self.params['W' + str(idx+1)] = weight_init_std * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        conv_pool_output_size = self.calculate_conv_pool_output_size(input_dim, conv_params, pool_params)
        
        self.params['W' + str(affine_start)] = weight_init_std * np.random.randn(conv_pool_output_size, hidden_size)
        self.params['b' + str(affine_start)] = np.zeros(hidden_size)
        self.params['W' + str(affine_start + 1)] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b' + str(affine_start + 1)] = np.zeros(output_size)

        # 生成层===========
        self.layers = []
        for idx, conv_param in enumerate(conv_params):
            self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                               conv_param['stride'], conv_param['pad']))
            self.conv_layer.append(len(self.layers) - 1)
            self.layers.append(Relu())
            if idx % 2: self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W' + str(affine_start)], self.params['b'+ str(affine_start)]))
        self.conv_layer.append(len(self.layers) - 1)
        self.layers.append(Relu())
        self.layers.append(Dropout(0.1))
        self.layers.append(Affine(self.params['W'+ str(affine_start + 1)], self.params['b'+ str(affine_start + 1)]))
        self.conv_layer.append(len(self.layers) - 1)
        self.layers.append(Dropout(0.1))

        self.last_layer = SoftmaxWithLoss()

    def calculate_conv_pool_output_size(self,input_size, conv_params, pool_params):
        pre_channel_num = input_size[0]
        output_height = input_size[1]
        output_width = input_size[2]
        
        for i, conv_param in enumerate(conv_params):
            filter_num = conv_param['filter_num']
            filter_size = conv_param['filter_size']
            pad = conv_param['pad']
            stride = conv_param['stride']
            
            output_height = (output_height + 2 * pad - filter_size) // stride + 1
            output_width = (output_width + 2 * pad - filter_size) // stride + 1

            if i % 2: 
                output_height = (output_height - pool_params['pool_size']) // pool_params['stride'] + 1
                output_width = (output_width - pool_params['pool_size']) // pool_params['stride'] + 1
        
        return filter_num * output_height * output_width
    
    def predict(self, x, train_flg=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = self.layers.copy()
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i, layer_idx in enumerate(self.conv_layer):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db
        return grads


    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, layer_idx in enumerate(self.conv_layer):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
