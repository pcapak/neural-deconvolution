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
import pickle

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
    #hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
#    print(first_two_rows)

os.chdir(os.getcwd()+"/xdc_iter1")
num_cols = 4
num_rows = data.field(0).shape[1]
ndata = np.zeros((num_rows, num_cols))
errors = np.zeros((num_rows, num_cols))
for i in range(num_cols):
    ndata[:, i] = data.field(i + 4)
    errors[:, i] = data.field(i + 8)
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

#bin_values = []
#with open('bin_values.txt', 'r') as filehandle:
#    lines = filehandle.read().splitlines()
#    for l in lines:
#        bin_values.append(float(l))

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


iters = [1, 10, 100, 1000, 10000]
times = {}
log_like = {}

from scipy.ndimage.filters import gaussian_filter

def perform_MCMC(xmean, xcovar, xamp1):
    tot_samp = 100000 * 100
    dim = xmean.shape[1]
    
    #figure out the number of gaussians in the model
    Ngauss = len(xamp1)
    
    samples = np.zeros([tot_samp,dim],dtype=np.double)
    
    #loop over the gaussians in the model
    sample_count = 0  #counter for number of samples
    for g in range(0,Ngauss):
        
        #set the number of samples proportional to the amplitude of this gaussian
        Gsamp = int(tot_samp*xamp1[g])       
        samples[sample_count:(sample_count+Gsamp),:] = np.random.multivariate_normal(xmean[g], xcovar[g], Gsamp)
        sample_count+=Gsamp
    g_min = np.amin(samples[:,0])
    r_min = np.amin(samples[:,1])
    i_min = np.amin(samples[:,2])
    z_min = np.amin(samples[:,3])
    g_max = np.amax(samples[:,0])
    r_max = np.amax(samples[:,1])
    i_max = np.amax(samples[:,2])
    z_max = np.amax(samples[:,3])
    
    g_diff = g_max-g_min
    r_diff = r_max-r_min
    i_diff = i_max-i_min
    z_diff = z_max-z_min
    
    pixel_num=301
    
    samples[:, 0] =  (samples[:, 0] - np.repeat([g_min], samples.shape[0])) * \
     np.repeat([pixel_num/g_diff], samples.shape[0])
    samples[:, 1] = (samples[:, 1] - np.repeat([r_min], samples.shape[0])) * \
     np.repeat([pixel_num/r_diff], samples.shape[0])
    samples[:, 2] = (samples[:, 2] - np.repeat([i_min], samples.shape[0])) * \
     np.repeat([pixel_num/i_diff], samples.shape[0])
    samples[:, 3] = (samples[:, 3] - np.repeat([z_min], samples.shape[0])) * \
     np.repeat([pixel_num/z_diff], samples.shape[0])
    #print(samples)
    return samples

def create_contours(data, pixel_num):
    image_yx = np.zeros([pixel_num, pixel_num],dtype=np.double)
    #print(data)
    for point in data:
        y = int(point[0])
        x = int(point[1])
        if x > 0 and x < pixel_num and y > 0 and y < pixel_num:
            image_yx[y, x] += 1
    # Normalize the probability distribution
    for i in range(image_yx.shape[0]):
        for j in range(image_yx.shape[1]):
            image_yx[i][j] /= data.shape[0]
    print(data.shape[0])
    image_yx = gaussian_filter(image_yx, sigma=3)
    return image_yx

    
pixel_num = 301
# Run extreme deconvolution for certain max iterations
for it in iters:
    
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
    l = extreme_deconvolution(ydata,ycovar,xamp1,xmean,xcovar,weight=weights,maxiter=it)
    t1 = time.time()
    xdc_time = t1 - t0
    print(str(it)+" iteration(s) done")
    times[it] = xdc_time
    log_like[it] = l
    timesfile = open('iters_time', 'wb') 
    pickle.dump(times, timesfile)                      
    timesfile.close() 
    llfile = open('iters_ll', 'wb') 
    pickle.dump(log_like, llfile)                      
    llfile.close() 
    
    # create contour plots
    samples = perform_MCMC(xmean, xcovar, xamp1)
    #print(samples)
    image_gr = create_contours(np.array([samples[:, 0], samples[:, 1]]).T, pixel_num)
    image_ri = create_contours(np.array([samples[:, 1], samples[:, 2]]).T, pixel_num)
    image_iz = create_contours(np.array([samples[:, 2], samples[:, 3]]).T, pixel_num)
    image_gz = create_contours(np.array([samples[:, 0], samples[:, 3]]).T, pixel_num)
    #print(np.sum(image_gr))
    fits.writeto('gr_iter=' + str(it) + '.fits',image_gr,overwrite=True)
    fits.writeto('ri_iter=' + str(it) + '.fits',image_ri,overwrite=True)
    fits.writeto('iz_iter=' + str(it) + '.fits',image_iz,overwrite=True)
    fits.writeto('gz_iter=' + str(it) + '.fits',image_gz,overwrite=True)



    


    
    
    
    

    
    