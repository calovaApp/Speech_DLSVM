
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io.wavfile
import tflearn
from python_speech_features import fbank
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
import matplotlib.pyplot as plt

# In[2]:

def mfsc(signal, samplerate):
    feat, energy = fbank(signal, samplerate)
    feat = np.log(feat)

    return feat


# #parah datanya walau length nya sama2 6 seconds, matrix length nya berbeda, jadi harus entah diganti datasetnya
# #atau di potong semua datanya supaya sama
# #ternyata kalo di potong pake udacity dengan length yang sama baru bisa bener ukuran size nya (shape)
# [fs1,signal1] = scipy.io.wavfile.read('commands/c10r5.wav')
# [fs2,signal2] = scipy.io.wavfile.read('commands/c10r6.wav')
# static1 = mfsc(signal1, fs1)
# static2 = mfsc(signal2, fs2)
# print(static1.shape)
# print(static2.shape)

# In[3]:

def get_delta(inp):
    delta = np.zeros(inp.shape)
    for i in xrange(inp[0,].size):
        coef = i
        if coef == 0:
            bef = inp[i,coef]
            #print('bef=0',bef)
        else :
            bef = inp[i,coef-1]
            #print('bef other 0',bef)
    
        if coef == (inp[0].size)-1:
            af = inp[i,coef]
            #print('af at 25',af)
        else :
            af = inp[i,coef+1]
            #print('af other 25', af)
        delta[i,coef]= (af-bef)/2.0
    return delta


# In[4]:

def reshape_inp(static, delta, ddelta):
    input_array=np.array([static,delta,ddelta])
    input_array = input_array.reshape(-1,300,26,3)
    return input_array


# In[23]:

sig_len = 300
sig_feat = 26
X = np.zeros((60,sig_len, sig_feat, 3), dtype='float32')
y = np.zeros(60)
count = 0

#digit 1 and labels
for j in range(10):
    for i in range(6):
        [fs,signal] = scipy.io.wavfile.read('commands/c'+str(j+1)+'r'+str(i+1)+'.wav')
        #print(signal)
        static = mfsc(signal, fs)
        #print(static.shape)
        delta = get_delta(static)
        ddelta = get_delta(delta)
        new_inp = reshape_inp(static, delta, ddelta)
        X[count] = np.array(new_inp)
        
        y[count] = j
        count += 1
print(X.shape)

plt.imshow(X[11])
print(len(X))
print(X[10])


# In[15]:

#splitting dataset supaya random
X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.2)

print(Y_test)


# In[7]:

# encode the Ys
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

#print(Y)
print(Y_test)


# In[10]:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3])

conv_1 = conv_2d(network, 32, 5, activation='relu', name='conv_1')
#pooling layer with filter size 2x2
network = max_pool_2d(conv_1, 2)
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')
network = max_pool_2d(conv_1, 2)
#FC layer with number of neurons 512
network = fully_connected(network, 512, activation='relu')
#dropout layer to prevent overfitting
network = dropout(network, 0.5)
#final FC with classifier : softmax
network = fully_connected(network, 2, activation='softmax')


# In[12]:

#calculate the accuracy
acc = Accuracy(name="Accuracy")
#define how network to be trained
#it uses crossentropy loss function, adam optimizer for gradient descent, learning rate 0.0005 
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_speech.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')


# Train using classifier
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=10, run_id='speech_cnn')


