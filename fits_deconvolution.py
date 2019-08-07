#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:43:51 2019

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
#    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    
os.chdir(os.getcwd()+"/xdc_t2")

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


num_cols = 4
num_rows = data.field(0).shape[1]

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
#print(ndata.shape)
#print(errors.shape)

#ydata = ndata[:100,:]
ydata = ndata
print(ydata)
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
print(weights)

#print(ycovar.shape)
ngauss = np.shape(neurons)[0]
#init_sigma = ycovar[0:ngauss]
#ngauss = 50
dx = 4

# Divide by a number close to ngauss, but less than it
xamp1 = np.ones(ngauss)/(ngauss - 80)
xamp2 = np.ones(ngauss)/(ngauss - 80)
#xmean should be filled with the positions of the neurons 
#xmean = np.array([np.ones(ngauss) * np.mean(ndata[0]), 
#                  np.ones(ngauss) * np.mean(ndata[1])]).T
xmean = neurons[0:ngauss, :]
xcovar = np.zeros([ngauss, dx, dx])
#print(np.shape(xcovar))
#xcovar = np.cov(neurons.T)
for i in range(ngauss):
    for j in range(dy):
        xcovar[i][j][j] = 0.05
# make a copy of initial xcovar
#init_sigma = xcovar.copy()
#neurons = xmean.copy()

#print("xmean \n" + str(xmean))
print(ydata.shape)
print(ycovar.shape)
print(xamp1.shape)
print(xmean.shape)
print(xcovar.shape)
l = extreme_deconvolution(ydata,ycovar,xamp1,xmean,xcovar,weight=weights)

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

print("log likelihood: "+str(l))
print("new xmean \n" + str(xmean))
print("new xamp \n" + str(xamp1))
print("diff in  xamp \n" + str(xamp1-xamp2))
print("new xcovar: \n" + str(xcovar))

sys.stdout = orig_stdout
f.close()

with open('new_xmeans.txt', 'w') as filehandle:
    for val in xmean:
        for comp in val:
            filehandle.write(str(comp)+' ')
        filehandle.write('\n')
        
with open('new_xcovars.txt', 'w') as filehandle:
    for index in range(xcovar.shape[0]):
        for i in range(dx):
            for j in range(dx):
                filehandle.write(str(xcovar[index][i][j])+' ')
        filehandle.write('\n')
        
with open('new_xamps.txt', 'w') as filehandle:
    for val in xamp1:
        filehandle.write(str(val)+'\n')
