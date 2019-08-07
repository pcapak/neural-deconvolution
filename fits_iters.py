#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:51:30 2019

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
import pickles

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    print(cols.info)
    first_two_rows = data[:2]
#    print(first_two_rows)
print(header)

if os.path.isdir("xdc_iter1"):
    shutil.rmtree("xdc_iter1")
os.mkdir("xdc_iter1")
os.chdir(os.getcwd()+"/xdc_iter1")

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

neurons = []
with open('neurons.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        neurons.append(a.astype(np.float))
neurons = np.array(neurons)

bin_values = []
with open('bin_values.txt', 'r') as filehandle:
    lines = filehandle.read().splitlines()
    for l in lines:
        bin_values.append(float(l))


iters = [1, 10, 100, 1000, 10000, 1e+5]
times = {}
log_like = {}



# Run extreme deconvolution for certain max iterations
for i in iters:
    
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
    l = extreme_deconvolution(ydata,ycovar,xamp1,xmean,xcovar,weight=weights,maxiter=i)
    t1 = time.time()
    xdc_time = t1 - t0
    print(str(i)+" iteration(s) done")
    iters[i] = xdc_time
    log_like[i] = l
    
    # create contour plots
    
    
    
    

    
    