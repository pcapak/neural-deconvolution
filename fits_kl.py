#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:43:52 2019

@author: MeganT
"""
from astropy.io import fits
import numpy.linalg as li
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
    
os.chdir(os.getcwd()+"/xdc_test")

num_cols = 4
num_rows = data.field(0).shape[1]

# We fill ndata with the true values of the flux
true_data = np.zeros((num_rows, num_cols))

for i in range(num_cols):
    true_data[:, i] = data.field(i)

# Normalize the true_data
for i in range(num_rows):
    total_flux = sum(true_data[i, :])
    true_data[i, :] = true_data[i, :] / total_flux

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

xmean = []
with open('new_xmeans.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        xmean.append(a.astype(np.float))
xmean = np.array(xmean)

dx = 4
xcovar = []
with open('new_xcovars.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        matrix = np.zeros([dx, dx])
        k = 0
        for i in range(dx):
            for j in range(dx):
                matrix[i][j] = a[k]
                k += 1
        xcovar.append(a.astype(np.float))
xcovar = np.array(xcovar)

xamp1 = []
with open('new_xamps.txt', 'r') as filehandle:
    lines = filehandle.read().splitlines()
    for l in lines:
        xamp1.append(float(l))

# Bins true distribution into neurons
samples = 2
data = np.zeros([num_rows*samples,4],dtype=np.float)
for i in range(num_rows):
    x = true_data[i,:]
    data[i*samples:(i+1)*samples, :] = np.asarray(
            [np.random.normal(x[0], 0,samples),
             np.random.normal(x[1], 0,samples),
             np.random.normal(x[2], 0, samples),
             np.random.normal(x[3], 0, samples)]).T


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
p, indexes = bin_data(data.T,neurons)
print('Binning Done')
p = p / np.sum(p)
d= 4
#init_sigma is 0
V = xcovar
# q is the estimated distribution
q = []
for v in neurons:
    s = 0
    for j in range(len(xmean)):
        diff = v - xmean[j]
        #print(xmean[j])
        #print(np.matmul(diff.T, li.inv(V[j])))
        scalar = np.matmul(np.matmul(diff.T, li.inv(V[j])), diff)
        gaussian =np.power((2*np.pi),(-d/2)) * np.power(li.det(V[j]),-1/2)* np.exp((-1/2)*scalar)
        #print(li.det(V[j]))
        s += xamp1[j] * gaussian
        #print(q)
    q.append(s)

