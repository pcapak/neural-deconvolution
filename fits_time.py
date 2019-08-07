#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:08:58 2019

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
import time

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
#    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    
num_cols = 4
num_rows = data.field(0).shape[1]
# Each data point is represented by a row

if os.path.isdir("4D_xdc_time_w"):
    shutil.rmtree("4D_xdc_time_w")
os.mkdir("4D_xdc_time_w")
os.chdir(os.getcwd()+"/4D_xdc_time_w")

# We fill ndata with the true values of the flux
true_data = np.zeros((num_rows, num_cols))

for i in range(num_cols):
    true_data[:, i] = data.field(i)

# Normalize the true_data
for i in range(num_rows):
    total_flux = sum(true_data[i, :])
    true_data[i, :] = true_data[i, :] / total_flux
    
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
    
thresholds = [0.10, 0.25, 0.50, 0.75, 0.9, 1, 2, 3]

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


def run_gng(data, i):
    # Training will slow down overtime and we increase number
    # of data samples for training
    #n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))
    n = int(i)
    #First argument was len(data)
    sampled_data_ids = np.random.choice(len(data), n)
    sampled_data = data[sampled_data_ids, :]
    gng.train(sampled_data, epochs=1)

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

for i in thresholds:
    print(i)
    labels = np.random.uniform(size = ndata.shape[0])
    if i > 1:
        new_ndata = ndata.copy()
        for j in range(i-1):
            new_ndata = np.concatenate((new_ndata, ndata))
            #print(new_ndata.shape)
    else:
        new_ndata = ndata[np.where(labels < i)]
    t0 = time.time()
    run_gng(new_ndata, new_ndata.shape[0] * 0.8)
    t1 = time.time()
    gng_time = t1 - t0
    nodes = gng.graph.nodes
    data = []
    for i in nodes:
        data.append(np.array(i.weight))
    data = np.squeeze(np.array(data))
    #print(data.shape)
    neurons = data
    
    t0 = time.time()
    bin_values, indexes = bin_data(new_ndata,neurons) 
    t1 = time.time()
    bin_time = t1 - t0
    
    ydata = ndata
    #print(ydata)
    dy = np.shape(ydata.T)[0]
    len_data = np.shape(ydata)[0]
    
    #ycovar should be filled with sigmas used to generate the 3D noise
    ycovar = np.zeros([len_data, dy, dy])
    weights = np.zeros(len_data)
    for i in range(len_data):
        sum_sigma = 0
        for j in range(dy):
            error_squared = np.square(errors[i][j])
            ycovar[i][j][j] = error_squared
            sum_sigma += error_squared
        weights[i] = 1/sum_sigma
    #print(weights)
    
    ngauss = np.shape(neurons)[0]
    dx = 4
    
    # Divide by a number close to ngauss, but less than it
    constant = ngauss / 5
    #print(constant)
    xamp1 = np.ones(ngauss)/(ngauss - constant)
    xamp2 = np.ones(ngauss)/(ngauss - constant)
    xmean = neurons[0:ngauss, :]
    xcovar = np.zeros([ngauss, dx, dx])
    #print(np.shape(xcovar))
    #xcovar = np.cov(neurons.T)
    for i in range(ngauss):
        xcovar[i][0][0] = 0.00044115
        xcovar[i][1][1] = 0.00074033
        xcovar[i][2][2] = 0.00216775
        xcovar[i][3][3] = 0.0073491
            
    t0 = time.time()
    l = extreme_deconvolution(ydata,ycovar,xamp1,xmean,xcovar,weight=weights)
    t1 = time.time()
    xdc_time = t1 - t0
    filename = "threshold_"+str(new_ndata.shape[0])
    with open(filename, 'w') as filehandle:
        filehandle.write("ndata: "+str(new_ndata.shape[0]) + '\n')
        filehandle.write("ngauss: "+str(ngauss) + '\n')
        filehandle.write("gng: "+str(gng_time) + '\n')
        filehandle.write("binning: "+str(bin_time) + '\n')
        filehandle.write("deconvolution: "+str(xdc_time) + '\n')
        
