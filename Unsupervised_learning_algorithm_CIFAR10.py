import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import time

import torch

dir = "./data/cifar-10-batches-mat"
filelist = ["data_batch_1.mat",
            "data_batch_2.mat",
            "data_batch_3.mat",
            "data_batch_4.mat",
            "data_batch_5.mat"]

"""
def draw_weights(synapses, Kx, Ky):
    yy = 0
    HM = np.zeros((32*Ky, 32*Kx, 3))
    for y in range(Ky):
        for x in range(Kx):
            R = synapses[yy]
            min_R = np.amin(R)
            if min_R < 0:
                R += abs(min_R)

            max_R = np.amax(R)
            R /= max_R
            single_img_reshaped = np.transpose(np.reshape(R, (3, 32, 32)), (1, 2, 0))

            HM[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32, :] = single_img_reshaped
            #yy += 1
    plt.clf()
    nc = np.amax(np.absolute(HM))
    im = plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    #fig = plt.figure(figsize=(12.9, 10))
    #fig.colorbar(im, ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    #fig.canvas.draw()
    plt.show()

"""
def draw_weights(synapses, Kx, Ky):
    yy = 0
    HM = np.zeros((32*Ky, 32*Kx, 3))
    for y in range(Ky):
        for x in range(Kx):
            #HM[y*32:(y+1)*32, x*32:(x+1)*32, :] = synapses[yy].reshape(32, 32, 3)
            synapse_tmp = np.zeros((32, 32, 3))
            synapse_tmp[:, :, 0] = synapses[yy, 0:1024].reshape(32, 32)
            synapse_tmp[:, :, 1] = synapses[yy, 1024:2048].reshape(32, 32)
            synapse_tmp[:, :, 2] = synapses[yy, 2048:3072].reshape(32, 32)
            min_synapse_tmp = np.amin(synapse_tmp)
            if min_synapse_tmp < 0:
                synapse_tmp -= min_synapse_tmp
            ratio = 255/np.amax(synapse_tmp)
            synapse_tmp *= ratio
            HM[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32, :] = synapse_tmp
            #HM[y*32:(y+1)*32, x*32:(x+1)*32, 0] = synapses[yy, 0:1024].reshape(32, 32)
            #HM[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32, 1] = synapses[yy, 1024:2048].reshape(32, 32)
            #HM[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32, 2] = synapses[yy, 2048:3072].reshape(32, 32)
            yy += 1


    HM = HM.astype(np.uint8)
    im = Image.fromarray(HM)
    im.show()


num_of_class = 10
N = 3072
num_of_set = 50000
train_data = np.zeros((0, N))
for file in filelist:
    mat = scipy.io.loadmat(dir + '/' + file)
    train_data = np.concatenate((train_data, mat['data']), axis=0)
train_data = train_data/255.0
print(train_data.shape)


# Standardization
R = train_data[:32 * 32]
G = train_data[32 * 32:32 * 32 * 2]
B = train_data[32 * 32 * 2:32 * 32 * 3]

mean_R = 0.4914
mean_G = 0.4822
mean_B = 0.4465

stdev_R = 0.2023
stdev_G = 0.1994
stdev_B = 0.2010

train_data[:32 * 32] = (R - mean_R) / stdev_R
train_data[32 * 32:32 * 32 * 2] = (G - mean_G) / stdev_G
train_data[32 * 32 * 2:32 * 32 * 3] = (B - mean_B) / stdev_B


learning_rate = 2e-2   # learning rate
Kx = 20
Ky = 20
num_of_hid = Kx*Ky   # number of hidden units that are displayed in Ky by Kx array
mu = 0.0
sigma = 1.0
epochs = 100     # number of epochs
batch_size = 100     # size of the minibatch
prec = 1e-30
delta = 0.1   # Strength of the anti-hebbian learning
p = 2.0       # Lebesgue norm of the weights
k = 2         # ranking parameter, must be integer that is bigger or equal than 2

#fig = plt.figure(figsize=(12.9, 10))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# move train data to cuda
train_data = torch.from_numpy(train_data).float().to(device)


synapses = (torch.randn(num_of_hid, N) * sigma + mu).to(device)

start_time = time.time()

for nep in range(epochs):
    eps = learning_rate * (1-nep/epochs)
    train_data = train_data[np.random.permutation(num_of_set), :]
    for i in range(num_of_set // batch_size):
        inputs = torch.transpose(train_data[i*batch_size:(i+1)*batch_size, :], 0, 1).to(device)
        sig = torch.sign(synapses).to(device)
        tot_input = torch.matmul(sig*torch.abs(synapses).pow_(p-1), inputs).to(device)

        y = torch.argsort(tot_input, dim=0).to(device)
        y1 = torch.zeros((num_of_hid, batch_size)).to(device)
        tmp = y[num_of_hid - 1, :]
        y1[y[num_of_hid-1, :], np.arange(batch_size)] = 1.0
        y1[y[num_of_hid-k], np.arange(batch_size)] = -delta
        xx = torch.sum(torch.mul(y1, tot_input), 1).to(device)

        ds = torch.matmul(y1, torch.transpose(inputs, 0, 1)) - torch.mul(xx.reshape(xx.shape[0],1).repeat(1, N), synapses).to(device)
        nc = torch.max(torch.abs(ds))
        if nc < prec:
            nc = prec
        synapses += eps * torch.div(ds, nc)
    #synapses = synapses.cpu().numpy()
    #draw_weights(synapses, Kx, Ky)
    #synapses = torch.from_numpy(synapses).float().to(device)
    print("epoch : ", nep)
    if (nep % 100) == 0:
        np.save("synapses/synapses_%d.npy" % nep, synapses.cpu().numpy())

np.save("synapses/synapses_%d.npy" % epochs, synapses.cpu().numpy())

print("--- %s seconds ---" % (time.time() - start_time))














