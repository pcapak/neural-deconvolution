#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:50:57 2019

@author: MeganT
"""

from astropy.io import fits
from extreme_deconvolution import extreme_deconvolution
import matplotlib.pyplot as plt
import shutil 
import os
import numpy as np
import math
import sys

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    print(cols.info)
    first_two_rows = data[:2]
#    print(first_two_rows)
print(header)

if os.path.isdir("xdc_test_noisy"):
    shutil.rmtree("xdc_test_noisy")
os.mkdir("xdc_test_noisy")
os.chdir(os.getcwd()+"/xdc_test_noisy")

num_cols = 4
num_rows = data.field(0).shape[1]
# Each data point is represented by a row

# We fill ndata with the true values of the flux
true_data = np.zeros((num_rows, num_cols))

for i in range(num_cols):
    true_data[:, i] = data.field(i)

# Normalize the true_data
for i in range(num_rows):
    total_flux = sum(true_data[i, :])
    true_data[i, :] = true_data[i, :] / total_flux
    
with open('true_data.txt', 'w') as filehandle:
    for val in true_data:
        for comp in val:
            filehandle.write(str(comp)+' ')
        filehandle.write('\n')
    
    
#print(total_flux)
#print(true_data.shape)
#print(ndata)

# Plot the true flux 
g_norm = true_data[:, 0] 
r_norm = true_data[:, 1] 
i_norm = true_data[:, 2] 
z_norm = true_data[:, 3] 

fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.plot(r_norm, g_norm)
ax1.set_title("Normalized G vs. Normalized R")
ax2.plot(i_norm, r_norm)
ax2.set_title("Normalized R vs. Normalized I")
ax3.plot(z_norm, i_norm)
ax3.set_title("Normalized I vs. Normalized Z")
ax4.plot(z_norm, g_norm)
ax4.set_title("Normalized G vs. Normalized Z")
plt.savefig("True Fluxes.png")
#plt.show()

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.plot(g_norm / r_norm, r_norm / i_norm)
ax1.set_title("G/R vs. R/I")
ax2.plot(r_norm / i_norm, i_norm / z_norm)
ax2.set_title("R/I vs. I/Z")
ax3.plot(g_norm / i_norm, i_norm / z_norm)
ax3.set_title("G/I vs. I/Z")
ax4.plot(g_norm / i_norm, r_norm / z_norm)
ax4.set_title("G/I vs. R/Z")
plt.savefig("Ratio of Fluxes.png")

# Read in the noisy data and their errors
ndata = np.zeros((num_rows, num_cols))
errors = np.zeros((num_rows, num_cols))
for i in range(num_cols):
    ndata[:, i] = data.field(i + 4)
    errors[:, i] = data.field(i + 8)
#print(ndata)

# Normalize the noisy data and errors by the same value
for i in range(num_rows):
    total_flux = sum(ndata[i, :])
    ndata[i, :] = ndata[i, :] / total_flux
    errors[i, :] = errors[i, :] / total_flux
#print(ndata)

g_norm = ndata[:, 0] 
r_norm = ndata[:, 1] 
i_norm = ndata[:, 2] 
z_norm = ndata[:, 3] 

fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.plot(r_norm, g_norm)
ax1.set_title("Normalized G vs. Normalized R")
ax2.plot(i_norm, r_norm)
ax2.set_title("Normalized R vs. Normalized I")
ax3.plot(z_norm, i_norm)
ax3.set_title("Normalized I vs. Normalized Z")
ax4.plot(z_norm, g_norm)
ax4.set_title("Normalized G vs. Normalized Z")
plt.savefig("Noisy Fluxes.png")
#plt.show()

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.plot(g_norm / r_norm, r_norm / i_norm)
ax1.set_title("G/R vs. R/I")
ax2.plot(r_norm / i_norm, i_norm / z_norm)
ax2.set_title("R/I vs. I/Z")
ax3.plot(g_norm / i_norm, i_norm / z_norm)
ax3.set_title("G/I vs. I/Z")
ax4.plot(g_norm / i_norm, r_norm / z_norm)
ax4.set_title("G/I vs. R/Z")
plt.savefig("Noisy Ratio of Fluxes.png")

# For running GNG on the true values.
    
#errors = np.zeros((num_rows, num_cols))
#errors[:,:] = 0.01
##print(errors.shape)
#samples = 1
#for i in range(num_rows):
#    x = true_data[i,:]
#    ndata[i*samples:(i+1)*samples, :] = np.asarray(
#            [np.random.normal(x[0], errors[i, 0],samples),
#             np.random.normal(x[1],errors[i, 1],samples),
#             np.random.normal(x[2], errors[i, 2], samples),
#             np.random.normal(x[3], errors[i, 3], samples)]).T
    
#print(true_data)
#print(ndata)
#print(errors)
    

from neupy import algorithms, utils
utils.reproducible()

#gng with 3 features, because the data is 3-D now
gng = algorithms.GrowingNeuralGas(
    n_inputs=4,
    n_start_nodes=2,

    shuffle_data=True,
    verbose=False,
    
    step=0.1,
    neighbour_step=0.001,
    
    max_edge_age=50,
    # Before this was 1000
    max_nodes=10000,
    
    # To get 5000 notes, this was 2
    # To get 554 nodes, this was 20
    # To get 112 nodes, this was 100
    n_iter_before_neuron_added=20,
    after_split_error_decay_rate=0.5,
    error_decay_rate=0.995,
    min_distance_for_update=0.2,
)

def run_gng(i):
    # Training will slow down overtime and we increase number
    # of data samples for training
    #n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))
    n = int(i)
    #First argument was len(data)
    sampled_data_ids = np.random.choice(len(ndata), n)
    sampled_data = ndata[sampled_data_ids, :]
    gng.train(sampled_data, epochs=1)
    
run_gng(ndata.shape[0] * 0.8)
nodes = gng.graph.nodes
neurons = []
for i in nodes:
    neurons.append(np.array(i.weight))
neurons = np.squeeze(np.array(neurons))
print(neurons.shape)

def dist(s, n):
    ans = 0
    for i in range(len(s)):
        ans += (s[i]-n[i])**2
    return math.sqrt(ans)

def bin_data(samples, neurons):
    output = np.zeros(neurons.shape[0])
    indexes = np.zeros(samples.shape[0])
    k = 0
    for s in samples:
        min_dist = sys.float_info.max
        min_index = -1
        for i in range(neurons.shape[0]):
            d = dist(s, np.array(neurons[i, :]))
            if d < min_dist:
                min_dist = d
                min_index = i
        output[min_index] = output[min_index] + 1
        indexes[k] = min_index
        k += 1
    return (output, indexes)

bin_values, indexes = bin_data(ndata,neurons)   
with open('neurons.txt', 'w') as filehandle:
    for neuron in neurons:
        for comp in neuron:
            filehandle.write(str(comp)+' ')
        filehandle.write('\n')

with open('bin_values.txt', 'w') as filehandle:
    for val in bin_values:
        filehandle.write(str(val)+'\n')

# Continue on with deconvolution if binning neurons doesn't take much time
#ydata = ndata
##print(ydata)
#dy = np.shape(ydata.T)[0]
#len_data = np.shape(ydata)[0]
#
##ycovar should be filled with sigmas used to generate the 3D noise
#ycovar = np.zeros([len_data, dy, dy])
#for i in range(len_data):
#    for j in range(dy):
#        ycovar[i][j][j] = errors[i][j]
##print(ycovar)
#ngauss = np.shape(neurons)[0]
##init_sigma = ycovar[0:ngauss]
##ngauss = 50
#dx = 4
#print(neurons)
## Divide by a number close to ngauss, but less than it
#xamp1 = np.ones(ngauss)/(ngauss - 40)
#xamp2 = np.ones(ngauss)/(ngauss - 40)
##xmean should be filled with the positions of the neurons 
##xmean = np.array([np.ones(ngauss) * np.mean(ndata[0]), 
##                  np.ones(ngauss) * np.mean(ndata[1])]).T
#xmean = neurons[0:ngauss, :]
#xcovar = np.zeros([ngauss, dx, dx])
##print(np.shape(xcovar))
##xcovar = np.cov(neurons.T)
#for i in range(ngauss):
#    for j in range(dx):
#        xcovar[i][j][j] = 0.05
## make a copy of initial xcovar
##init_sigma = xcovar.copy()
##neurons = xmean.copy()
#
##print("xmean \n" + str(xmean))
##print(ydata.shape)
##print(ycovar.shape)
##print(xamp1.shape)
##print(xmean.shape)
##print(xcovar.shape)
#l = extreme_deconvolution(ydata,ycovar,xamp1,xmean,xcovar)
#
#orig_stdout = sys.stdout
#f = open('out.txt', 'w')
#sys.stdout = f
#
#print("log likelihood: "+str(l))
#print("new xmean \n" + str(xmean))
#print("new xamp \n" + str(xamp1))
#print("diff in  xamp \n" + str(xamp1-xamp2))
#print("new xcovar: \n" + str(xcovar))
#
#sys.stdout = orig_stdout
#f.close()
#
#with open('new_xmeans.txt', 'w') as filehandle:
#    for val in xmean:
#        for comp in val:
#            filehandle.write(str(comp)+' ')
#        filehandle.write('\n')


