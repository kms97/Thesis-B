# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:47:18 2018

@author: Kavi
"""

#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import timeit
#import pyrenn as prn
from pyrenn import train_LM, NNOut, CreateNN, saveNN, loadNN
#import pyrenn_c as prn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#train_set, valid_set, test_set = pickle.load( open( "mnist.pkl", "rb" ),encoding='latin1')
train_set = mnist.train
train_images = train_set.images.T
train_labels = (train_set.labels.T)
test_set = mnist.test
test_images = test_set.images.T
test_labels = (test_set.labels.T)

print ('x_train Examples Loaded = ' + str(train_images.shape))
print ('y_train Examples Loaded = ' + str(train_labels.shape))
print ('x_train Examples Loaded = ' + str(test_images.shape))
print ('y_train Examples Loaded = ' + str(test_labels.shape))

net = CreateNN([28*28,10,10])
batch_size = 100
number_of_batches=100
for i in range(number_of_batches):
    r = np.random.randint(0,55000-batch_size)
    X = train_images[:,r:r+batch_size]
    Y = train_labels[:,r:r+batch_size]
    start_time = timeit.default_timer()
    #Train NN with training data Ptrain=input and Ytrain=target
    #Set maximum number of iterations k_max
    #Set termination condition for Error E_stop
    #The Training will stop after k_max iterations or when the Error <=E_stop
    net = train_LM(X,Y,net,verbose=False,k_max=3,E_stop=1e-4)
    end_time = timeit.default_timer()
    print("Time Taken: ", (end_time-start_time))
    print('Batch No. ',i,' of ',number_of_batches)


#test_set = mnist.test
num = 10000
#for i in range(num):
P_ = test_images[:,0:num]
L_ = test_labels[:,0:num]
Y_ = NNOut(P_,net)
correct = 0
for i in range(num):	
	y_ = np.argmax(Y_[:,i])
	l_ = np.argmax(L_[:,i])
	if y_ == l_:
		correct = correct+1
print(correct)
#saveNN(net,"mnist_net_int32.csv")
#idx = np.random.randint(0,5000-9)
##testx = test_set.images.T
#P_ = test_images[:,idx:idx+9]
#Y_ = NNOut(P_,net)
##
#fig = plt.figure(figsize=[11,7])
#gs = mpl.gridspec.GridSpec(3,3)
#
#for i in range(9):
#
#    ax = fig.add_subplot(gs[i])
#
#    y_ = np.argmax(Y_[:,i]) #find index with highest value in NN output
#    p_ = P_[:,i].reshape(28,28) #Convert input data for plotting
#
#    ax.imshow(p_) #plot input data
#    ax.set_xticks([])
#    ax.set_yticks([])
#    ax.set_title(str(y_), fontsize=18)
#
#plt.show()
