#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:47:06 2019

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
    
os.chdir(os.getcwd()+"/xdc_t4")
num_cols = 4
num_rows = data.field(0).shape[1]

ndata = np.zeros((num_rows, num_cols))
errors = np.zeros((num_rows, num_cols))
for i in range(num_cols):
    ndata[:, i] = data.field(i + 4)
    errors[:, i] = data.field(i + 8)
    
for i in range(num_rows):
    total_flux = sum(ndata[i, :])
    ndata[i, :] = ndata[i, :] / total_flux
    errors[i, :] = errors[i, :] / total_flux
    
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
#print(neurons)

bin_values = []
with open('bin_values.txt', 'r') as filehandle:
    lines = filehandle.read().splitlines()
    for l in lines:
        bin_values.append(float(l))
#print(bin_values)

xmean = []
with open('new_xmeans.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        xmean.append(a.astype(np.float))
xmean = np.array(xmean)
#print(xmean)


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
        xcovar.append(matrix.astype(np.float))
xcovar = np.array(xcovar)

xamp1 = []
with open('new_xamps.txt', 'r') as filehandle:
    lines = filehandle.read().splitlines()
    for l in lines:
        xamp1.append(float(l))
xamp1 = np.array(xamp1)
#print(xamp1)

# Bins true distribution into neurons
samples = 2
data = np.zeros([num_rows*samples,4],dtype=np.float)
#sigma is 0 for all dimensions
for i in range(num_rows):
    x = true_data[i,:]
    data[i*samples:(i+1)*samples, :] = np.asarray(
            [np.random.normal(x[0], 0,samples),
             np.random.normal(x[1], 0,samples),
             np.random.normal(x[2], 0, samples),
             np.random.normal(x[3], 0, samples)]).T

    
# Apply MCMC to sample the XDC probability
tot_samp = 100000 * 100
dim = xmean.shape[1]

#figure out the number of gaussians in the model
Ngauss = len(xamp1)

samples = np.zeros([tot_samp,dim],dtype=np.double)

#loop over the gaussians in the model
sample_count = 0  #counter for number of samples
count_samples = np.zeros(Ngauss)
for g in range(0,Ngauss):
    
    #set the number of samples proportional to the amplitude of this gaussian
    Gsamp = int(tot_samp*xamp1[g])
    count_samples[g] = Gsamp
    
    samples[sample_count:(sample_count+Gsamp),:] = np.random.multivariate_normal(xmean[g], xcovar[g], Gsamp)
    sample_count+=Gsamp

count_samples = np.asarray(count_samples, np.int32)

# Plot the contours
V = xcovar
xamp = xamp1
old_samples = samples.copy()
#g_factor = 50 / np.mean(samples[:, 0])
#r_factor = 50 / np.mean(samples[:, 1])
#i_factor = 50 / np.mean(samples[:, 2])
#z_factor = 50 / np.mean(samples[:, 3])
#samples[:, 0] *= np.repeat([g_factor], samples.shape[0], axis = 0)
#samples[:, 1] *= np.repeat([r_factor], samples.shape[0], axis = 0)
#samples[:, 2] *= np.repeat([i_factor], samples.shape[0], axis = 0)
#samples[:, 3] *= np.repeat([z_factor], samples.shape[0], axis = 0)
#print(samples[:,0])
#g_diff = np.amax(samples[:,0]) - np.amin(samples[:,0])
#r_diff = np.amax(samples[:,1]) - np.amin(samples[:,1])
#i_diff = np.amax(samples[:,2]) - np.amin(samples[:,2])
#z_diff = np.amax(samples[:,3]) - np.amin(samples[:,3])
g_min = np.amin(samples[:,0])
r_min = np.amin(samples[:,1])
i_min = np.amin(samples[:,2])
z_min = np.amin(samples[:,3])
g_max = np.amax(samples[:,0])
r_max = np.amax(samples[:,1])
i_max = np.amax(samples[:,2])
z_max = np.amax(samples[:,3])

g_min_u = np.amin(old_samples[:,0])
r_min_u = np.amin(old_samples[:,1])
i_min_u = np.amin(old_samples[:,2])
z_min_u = np.amin(old_samples[:,3])
g_max_u = np.amax(old_samples[:,0])
r_max_u = np.amax(old_samples[:,1])
i_max_u = np.amax(old_samples[:,2])
z_max_u = np.amax(old_samples[:,3])

#print(g_min,r_min,i_min,z_min,g_max,r_max,i_max,z_max)
#g_min = 0.0
#r_min = 0.0
#i_min = 0.15
#z_min = 0.15
#g_max = 0.4
#r_max = 0.3
#i_max = 0.4
#z_max = 0.75

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

image_gr = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_ri = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_iz = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_gz = np.zeros([pixel_num, pixel_num],dtype=np.double)

for sample in samples:
    g = int(sample[0])
    r = int(sample[1])
    i = int(sample[2])
    z = int(sample[3])
    if g > 0 and g < pixel_num and r > 0 and r < pixel_num:
        image_gr[g, r] += 1
    if r > 0 and r < pixel_num and i > 0 and i < pixel_num:
        image_ri[r, i] += 1
    if i > 0 and i < pixel_num and z > 0 and z < pixel_num:
        image_iz[i, z] += 1
    if g > 0 and g < pixel_num and z > 0 and z < pixel_num:
        image_gz[g, z] += 1

g_grmin = pixel_num
g_grmax = -1
r_grmin = pixel_num
r_grmax = -1
for i in range(image_gr.shape[0]):
    for j in range(image_gr.shape[1]):
        image_gr[i][j] /= samples.shape[0]
        if image_gr[i][j] != 0:
            if i < g_grmin:
                g_grmin = i
            if j < r_grmin:
                r_grmin = j
            if i > g_grmax:
                g_grmax = i
            if j > r_grmax:
                r_grmax = j    
r_rimin = pixel_num
r_rimax = -1
i_rimin = pixel_num
i_rimax = -1     
for i in range(image_ri.shape[0]):
    for j in range(image_ri.shape[1]):
        image_ri[i][j] /= samples.shape[0]
        if image_ri[i][j] != 0:
            if i < r_rimin:
                r_rimin = i
            if j < i_rimin:
                i_rimin = j
            if i > r_rimax:
                r_rimax = i
            if j > i_rimax:
                i_rimax = j   
i_izmin = pixel_num
i_izmax = -1
z_izmin = pixel_num
z_izmax = -1 
for i in range(image_iz.shape[0]):
    for j in range(image_iz.shape[1]):
        image_iz[i][j] /= samples.shape[0]      
        if image_iz[i][j] != 0:
            if i < i_izmin:
                i_izmin = i
            if j < z_izmin:
                z_izmin = j
            if i > i_izmax:
                i_izmax = i
            if j > z_izmax:
                z_izmax = j   
g_gzmin = pixel_num
g_gzmax = -1
z_gzmin = pixel_num
z_gzmax = -1 
for i in range(image_gz.shape[0]):
    for j in range(image_gz.shape[1]):
        image_gz[i][j] /= samples.shape[0]
        if image_gz[i][j] != 0:
            if i < g_gzmin:
                g_gzmin = i
            if j < z_gzmin:
                z_gzmin = j
            if i > g_gzmax:
                g_gzmax = i
            if j > z_gzmax:
                z_gzmax = j   
        
#image_gr[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_gr.shape))
#image_ri[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_ri.shape))
#image_iz[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_iz.shape))
#image_gz[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_gz.shape))

from scipy.ndimage.filters import gaussian_filter

image_gr = gaussian_filter(image_gr, sigma=3)
fits.writeto('gr.fits',image_gr,overwrite=True)
image_ri = gaussian_filter(image_ri, sigma=3)
fits.writeto('ri.fits',image_ri,overwrite=True)
image_iz = gaussian_filter(image_iz, sigma=3)
fits.writeto('iz.fits',image_iz,overwrite=True)
image_gz = gaussian_filter(image_gz, sigma=3)
fits.writeto('gz.fits',image_gz,overwrite=True)

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
levels_gr=[1e-8, 1.16e-6,3.65e-6,9.0e-6,2.06e-5,4.54e-5,9.9e-5,2.14e-4,4.63e-4,1e-3]
levels_ri=[2.33043e-06,7.3512e-06,1.81681e-05,4.14725e-05,9.16801e-05, \
           0.000199849,0.000432893,0.00093497,0.00201666]
levels_iz=[2.06444e-06,6.51214e-06,1.60944e-05,3.67388e-05,8.12159e-05, \
           0.000177039,0.000383483,0.000828253,0.00178648]
levels_gz=[2.21448e-06,6.98544e-06,1.72642e-05,3.9409e-05,8.71186e-05, \
           0.000189906,0.000411354,0.00088845,0.00191632]
#fig.tight_layout()
ax1.contour(image_gr, levels_gr)
interval = 100

#i_ticks = np.around(np.linspace(i_min_u, i_max_u, len(ind)),2)
#z_ticks = np.around(np.linspace(z_min_u, z_max_u, len(ind)),2)

#ax1.set_xlim([0.018689261934454145, 0.30570522434917147])
#ax1.set_ylim([-0.02498731864164616, 0.37520237750109764])
ax1.set_xlabel("R")
ax1.set_ylabel("G")
x_ind = np.arange(start = r_grmin, stop = r_grmax, step=interval)
y_ind = np.arange(start = g_grmin, stop = g_grmax, step=interval)
r_ticks = np.around(np.linspace(r_min_u, r_max_u, len(x_ind)),2)
g_ticks = np.around(np.linspace(g_min_u, g_max_u, len(y_ind)),2)
ax1.set_xticks(x_ind)
ax1.set_yticks(y_ind)
ax1.set_xticklabels(r_ticks)
ax1.set_yticklabels(g_ticks)
ax1.set_xlim(r_grmin, r_grmax)
ax1.set_ylim(g_grmin, g_grmax)
ax1.set_title("Normalized G vs. Normalized R")

ax2.contour(image_ri, levels_ri)
#ax2.set_xlim([0.168327331025541, 0.3818603775513799])
#ax2.set_ylim([0.01591015407155072, 0.3084843322120745])
ax2.set_xlabel("I")
ax2.set_ylabel("R")
x_ind = np.arange(start = i_rimin, stop = i_rimax, step=interval)
y_ind = np.arange(start = r_rimin, stop = r_rimax, step=interval)
i_ticks = np.around(np.linspace(i_min_u, i_max_u, len(x_ind)),2)
r_ticks = np.around(np.linspace(r_min_u, r_max_u, len(y_ind)),2)
ax2.set_xticks(x_ind)
ax2.set_yticks(y_ind)
ax2.set_xticklabels(i_ticks)
ax2.set_yticklabels(r_ticks)
ax2.set_xlim(i_rimin, i_rimax)
ax2.set_ylim(r_rimin, r_rimax)
ax2.set_title("Normalized R vs. Normalized I")

ax3.contour(image_iz, levels_iz)
#ax3.set_xlim([0.10870964061586308, 0.7513928898988901])
#ax3.set_ylim([0.16554822316263784, 0.38463948541428344])
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
x_ind = np.arange(start = z_izmin, stop = z_izmax, step=interval)
y_ind = np.arange(start = i_izmin, stop = i_izmax, step=interval)
z_ticks = np.around(np.linspace(z_min_u, z_max_u, len(x_ind)),2)
i_ticks = np.around(np.linspace(i_min_u, i_max_u, len(y_ind)),2)
ax3.set_xticks(x_ind)
ax3.set_yticks(y_ind)
ax3.set_xticklabels(z_ticks)
ax3.set_yticklabels(i_ticks)
ax3.set_xlim(z_izmin, z_izmax)
ax3.set_ylim(i_izmin, i_izmax)
ax3.set_title("Normalized I vs. Normalized Z")

ax4.contour(image_gz,levels_gz)
#ax4.set_xlim([0.10870964061586297, 0.75139288989889])
#ax4.set_ylim([-0.024987318641646213, 0.3752023775010975])
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
x_ind = np.arange(start = z_gzmin, stop = z_gzmax, step=interval)
y_ind = np.arange(start = g_gzmin, stop = g_gzmax, step=interval)
z_ticks = np.around(np.linspace(z_min_u, z_max_u, len(x_ind)),2)
g_ticks = np.around(np.linspace(g_min_u, g_max_u, len(y_ind)),2)
ax4.set_xticks(x_ind)
ax4.set_yticks(y_ind)
ax4.set_xticklabels(z_ticks)
ax4.set_yticklabels(g_ticks)
ax4.set_xlim(z_gzmin, z_gzmax)
ax4.set_ylim(g_gzmin, g_gzmax)
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("Scaled and Pixelated")
plt.savefig("Scaled_And_Pixelated.png")


# Plot contours of true fluxes for comparison
true_copy = true_data.copy()
#g_min = np.amin(true_copy[:,0])
#r_min = np.amin(true_copy[:,1])
#i_min = np.amin(true_copy[:,2])
#z_min = np.amin(true_copy[:,3])
#g_max = np.amax(true_copy[:,0])
#r_max = np.amax(true_copy[:,1])
#i_max = np.amax(true_copy[:,2])
#z_max = np.amax(true_copy[:,3])


#g_min_u = np.amin(true_copy[:,0])
#r_min_u = np.amin(true_copy[:,1])
#i_min_u = np.amin(true_copy[:,2])
#z_min_u = np.amin(true_copy[:,3])
#g_max_u = np.amax(true_copy[:,0])
#r_max_u = np.amax(true_copy[:,1])
#i_max_u = np.amax(true_copy[:,2])
#z_max_u = np.amax(true_copy[:,3])

#print(g_min,r_min,i_min,z_min,g_max,r_max,i_max,z_max)
#g_min = 0.0
#r_min = 0.0
#i_min = 0.15
#z_min = 0.15
#g_max = 0.4
#r_max = 0.3
#i_max = 0.4
#z_max = 0.75

#g_diff = g_max-g_min
#r_diff = r_max-r_min
#i_diff = i_max-i_min
#z_diff = z_max-z_min

pixel_num=301

true_copy[:, 0] =  (true_copy[:, 0] - np.repeat([g_min], true_copy.shape[0])) * \
 np.repeat([pixel_num/g_diff], true_copy.shape[0])
true_copy[:, 1] = (true_copy[:, 1] - np.repeat([r_min], true_copy.shape[0])) * \
 np.repeat([pixel_num/r_diff], true_copy.shape[0])
true_copy[:, 2] = (true_copy[:, 2] - np.repeat([i_min], true_copy.shape[0])) * \
 np.repeat([pixel_num/i_diff], true_copy.shape[0])
true_copy[:, 3] = (true_copy[:, 3] - np.repeat([z_min], true_copy.shape[0])) * \
 np.repeat([pixel_num/z_diff], true_copy.shape[0])

image_gr = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_ri = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_iz = np.zeros([pixel_num, pixel_num],dtype=np.double)
image_gz = np.zeros([pixel_num, pixel_num],dtype=np.double)

for sample in true_copy:
    g = int(sample[0])
    r = int(sample[1])
    i = int(sample[2])
    z = int(sample[3])
    if g > 0 and g < pixel_num and r > 0 and r < pixel_num:
        image_gr[g, r] += 1
    if r > 0 and r < pixel_num and i > 0 and i < pixel_num:
        image_ri[r, i] += 1
    if i > 0 and i < pixel_num and z > 0 and z < pixel_num:
        image_iz[i, z] += 1
    if g > 0 and g < pixel_num and z > 0 and z < pixel_num:
        image_gz[g, z] += 1

for i in range(image_gr.shape[0]):
    for j in range(image_gr.shape[1]):
        image_gr[i][j] /= true_copy.shape[0]
        
for i in range(image_ri.shape[0]):
    for j in range(image_ri.shape[1]):
        image_ri[i][j] /= true_copy.shape[0]

for i in range(image_iz.shape[0]):
    for j in range(image_iz.shape[1]):
        image_iz[i][j] /= true_copy.shape[0]

for i in range(image_gz.shape[0]):
    for j in range(image_gz.shape[1]):
        image_gz[i][j] /= true_copy.shape[0]
        
#image_gr[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_gr.shape))
#image_ri[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_ri.shape))
#image_iz[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_iz.shape))
#image_gz[:,:] /= np.repeat(np.array([samples.shape[0]]), np.array(image_gz.shape))

image_gr = gaussian_filter(image_gr, sigma=3)
#fits.writeto('gr.fits',image_gr,overwrite=True)
image_ri = gaussian_filter(image_ri, sigma=3)
#fits.writeto('ri.fits',image_ri,overwrite=True)
image_iz = gaussian_filter(image_iz, sigma=3)
#fits.writeto('iz.fits',image_iz,overwrite=True)
image_gz = gaussian_filter(image_gz, sigma=3)
#fits.writeto('gz.fits',image_gz,overwrite=True)

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
levels=[1e-8, 1.16e-6,3.65e-6,9.0e-6,2.06e-5,4.54e-5,9.9e-5,2.14e-4,4.63e-4,1e-3]
#fig.tight_layout()
ax1.contour(image_gr, levels_gr)
interval = 100
#ind = np.arange(pixel_num, step=interval)

#ax1.set_xlim([0.018689261934454145, 0.30570522434917147])
#ax1.set_ylim([-0.02498731864164616, 0.37520237750109764])
ax1.set_xlabel("R")
ax1.set_ylabel("G")
#ax1.set_xticks(ind)
#ax1.set_yticks(ind)
x_ind = np.arange(start = r_grmin, stop = r_grmax, step=interval)
y_ind = np.arange(start = g_grmin, stop = g_grmax, step=interval)
r_ticks = np.around(np.linspace(r_min_u, r_max_u, len(x_ind)),2)
g_ticks = np.around(np.linspace(g_min_u, g_max_u, len(y_ind)),2)
ax1.set_xticks(x_ind)
ax1.set_yticks(y_ind)
ax1.set_xlim(r_grmin, r_grmax)
ax1.set_ylim(g_grmin, g_grmax)
ax1.set_xticklabels(r_ticks)
ax1.set_yticklabels(g_ticks)
ax1.set_title("Normalized G vs. Normalized R")

ax2.contour(image_ri, levels_ri)
#ax2.set_xlim([0.168327331025541, 0.3818603775513799])
#ax2.set_ylim([0.01591015407155072, 0.3084843322120745])
ax2.set_xlabel("I")
ax2.set_ylabel("R")
#ax2.set_xticks(ind)
#ax2.set_yticks(ind)
x_ind = np.arange(start = i_rimin, stop = i_rimax, step=interval)
y_ind = np.arange(start = r_rimin, stop = r_rimax, step=interval)
i_ticks = np.around(np.linspace(i_min_u, i_max_u, len(x_ind)),2)
r_ticks = np.around(np.linspace(r_min_u, r_max_u, len(y_ind)),2)
ax2.set_xticks(x_ind)
ax2.set_yticks(y_ind)
ax2.set_xticklabels(i_ticks)
ax2.set_yticklabels(r_ticks)
ax2.set_xlim(i_rimin, i_rimax)
ax2.set_ylim(r_rimin, r_rimax)
ax2.set_title("Normalized R vs. Normalized I")

ax3.contour(image_iz, levels_iz)
#ax3.set_xlim([0.10870964061586308, 0.7513928898988901])
#ax3.set_ylim([0.16554822316263784, 0.38463948541428344])
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
#ax3.set_xticks(ind)
#ax3.set_yticks(ind)
x_ind = np.arange(start = z_izmin, stop = z_izmax, step=interval)
y_ind = np.arange(start = i_izmin, stop = i_izmax, step=interval)
z_ticks = np.around(np.linspace(z_min_u, z_max_u, len(x_ind)),2)
i_ticks = np.around(np.linspace(i_min_u, i_max_u, len(y_ind)),2)
ax3.set_xticks(x_ind)
ax3.set_yticks(y_ind)
ax3.set_xticklabels(z_ticks)
ax3.set_yticklabels(i_ticks)
ax3.set_xlim(z_izmin, z_izmax)
ax3.set_ylim(i_izmin, i_izmax)
ax3.set_title("Normalized I vs. Normalized Z")

ax4.contour(image_gz,levels_gz)
#ax4.set_xlim([0.10870964061586297, 0.75139288989889])
#ax4.set_ylim([-0.024987318641646213, 0.3752023775010975])
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
#ax4.set_xticks(ind)
#ax4.set_yticks(ind)
x_ind = np.arange(start = z_gzmin, stop = z_gzmax, step=interval)
y_ind = np.arange(start = g_gzmin, stop = g_gzmax, step=interval)
z_ticks = np.around(np.linspace(z_min_u, z_max_u, len(x_ind)),2)
g_ticks = np.around(np.linspace(g_min_u, g_max_u, len(y_ind)),2)
ax4.set_xticks(x_ind)
ax4.set_yticks(y_ind)
ax4.set_xticklabels(z_ticks)
ax4.set_yticklabels(g_ticks)
ax4.set_xlim(z_gzmin, z_gzmax)
ax4.set_ylim(g_gzmin, g_gzmax)
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("True Fluxes Scaled and Pixelated")
plt.savefig("Scaled_And_Pixelated_True.png")

        

#image_gr = np.zeros([g_diff, r_diff],dtype=np.double)
#image_ri = np.zeros([r_diff, i_diff],dtype=np.double)
#image_iz = np.zeros([i_diff, z_diff],dtype=np.double)
#image_gz = np.zeros([g_diff, z_diff,],dtype=np.double)

#d=4
##factor_arr = np.repeat([20], d, axis = 0)
#count = 0
##print(count_samples)
#for i in range(len(count_samples)):
#    num_samples = count_samples[i]
#    for j in range(count, count + num_samples):
#        #diff = bin_xmean[i][j]-xmean[i]
#        #print(samples[j])
#        diff = old_samples[j] - xmean[i]
#        scalar = np.matmul(np.matmul(diff.T, li.inv(V[i])), diff)
#        #gaussian =np.power((2*np.pi),(-d/2)) * np.power(li.det(V[j]),-1/2)* np.exp((-1/2)*scalar)
#        gaussian = (2*np.pi)**(-d/2) * (li.det(V[i]))**(-1/2)* np.exp((-1/2)*scalar)
#        (g, r, i, z) = np.asarray(samples[j, :], np.int32)
##        print(g)
#        image_gr[g - g_min, r - r_min] += xamp[i] * gaussian
#        image_ri[r - r_min, i - i_min] += xamp[i] * gaussian
#        image_iz[i - i_min, z - z_min] += xamp[i] * gaussian
#        image_gz[g - g_min, z - z_min] += xamp[i] * gaussian
#    count += num_samples


