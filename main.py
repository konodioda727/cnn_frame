from common.trainer import Trainer
from common.functions import softmax
from dataset.load_data import load
from matplotlib import pyplot as plt
import numpy as np
from nets.conv_gene import FlexibleConvNet

save_file = '/dataset/dataset.pkl'
labels = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def main():
    train_set, test_set = load(save_file, flatten=False)  
    X_train, y_train = train_set 
    X_test, y_test = test_set
    model = FlexibleConvNet(conv_params=[
        {'filter_num':20, 'filter_size':5, 'pad':0, 'stride':1},
    ], hidden_size=100)
    
    # model.load_params()
    trainer = Trainer(model, X_train, y_train, X_test, y_test, optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000,optimizer='Adam', epochs=1, mini_batch_size=1000)
    trainer.train()
    trainer.show_loss()
    # plt.figure(figsize=(12, 8))
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # img = X_train[4]
    # plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)
    # plt.show()
    # predict = model.predict(img.reshape(1,1,28,28), train_flg=False)[0]
    # print(softmax(predict))
    # print(labels[np.argmax(softmax(predict))])
    
if __name__ == "__main__":
    main()