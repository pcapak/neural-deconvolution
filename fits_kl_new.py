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

g_norm = ndata[:, 0] 
r_norm = ndata[:, 1] 
i_norm = ndata[:, 2] 
z_norm = ndata[:, 3] 

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.scatter(r_norm, g_norm)
gr_xlim = ax1.get_xlim()
gr_ylim = ax1.get_ylim()
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(i_norm, r_norm)
ri_xlim = ax2.get_xlim()
ri_ylim = ax2.get_ylim()
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(z_norm, i_norm)
iz_xlim = ax3.get_xlim()
iz_ylim = ax3.get_ylim()
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(z_norm, g_norm)
gz_xlim = ax4.get_xlim()
gz_ylim = ax4.get_ylim()
ax4.set_title("Normalized G vs. Normalized Z")
plt.savefig("Noisy Fluxes Unscaled.png")

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
# xcovar values to try
print(np.mean(np.array(xcovar[:][0][0])))
print(np.mean(np.array(xcovar[:][1][1])))
print(np.mean(np.array(xcovar[:][2][2])))
print(np.mean(np.array(xcovar[:][3][3])))

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

p = []
if (os.path.isfile('true_bin_values.txt')):
    with open('true_bin_values.txt', 'r') as filehandle:
        lines = filehandle.readlines()
        for val in lines:
            p.append(val)
    p = np.array(p).astype(np.float)
else:
    p, indexes = bin_data(data, neurons)
    with open('true_bin_values.txt', 'w') as filehandle:
        for val in p:
            filehandle.write(str(val)+'\n')
    print('Binning Done')
p = p / np.sum(p)
print(p)
d= 4
#init_sigma is 0
V = xcovar

# q is the estimated distribution
q = []
for v in neurons:
    s = 0
    for j in range(len(xmean)):
        diff = v - xmean[j]
        scalar = np.matmul(np.matmul(diff.T, li.inv(V[j])), diff)
        gaussian =(2*np.pi)**(-d/2) * li.det(V[j])**(-1/2) * np.exp(-scalar/2)
        s += xamp1[j] * gaussian 
    q.append(s)


# Normalization of the estimated distribution
def apply_norm(est):
    est = est / np.sum(est)
    new_q = np.where(est < np.amin(p[np.where(p != 0)]), 0, est)
#    print(new_q)
    return new_q

def kl(p, q):

    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    
    
    Parameters
    
    ----------
    
    p, q : array-like, dtype=float, shape=n
    
    Discrete probability distributions.
    
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    arr = np.where(q!=0, p/ q,1)
    return np.sum(np.where(p!=0, p* np.log(arr),0))

prob = kl(p, apply_norm(q))
print(prob)

with open('noiseless_distr.txt', 'w') as filehandle:
    for val in p:
        filehandle.write(str(val)+'\n')

with open('estimated_distr.txt', 'w') as filehandle:
    for val in q:
        filehandle.write(str(val)+'\n')

    
with open('kl_results.txt', 'w') as filehandle:
    filehandle.write(str(prob)+'\n')
     

# Generate noisy data from the new xmeans 

samples = np.asarray(xamp1 * 99999)
samples = samples.astype(np.int32)
gen_ndata = np.zeros([sum(samples), xmean.shape[1]])
k = 0 
for i in range(xmean.shape[0]):
    noise = errors
    #print(xcovar[i])
    gen_ndata[k:k+samples[i], :] = np.asarray(
            [np.random.normal(xmean[i][0], noise[i][0],samples[i]),
             np.random.normal(xmean[i][1], noise[i][1],samples[i]),
             np.random.normal(xmean[i][2], noise[i][2], samples[i]),
             np.random.normal(xmean[i][3], noise[i][3], samples[i])]).T
    k += samples[i]
    


g_norm = gen_ndata[:, 0] 
r_norm = gen_ndata[:, 1] 
i_norm = gen_ndata[:, 2] 
z_norm = gen_ndata[:, 3] 

# Plot the Generated Noisy Data
plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
fig.tight_layout()
ax1.scatter(r_norm, g_norm)
ax1.set_xlim(gr_xlim)
ax1.set_ylim(gr_ylim)
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(i_norm, r_norm)
ax2.set_xlim(ri_xlim)
ax2.set_ylim(ri_ylim)
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(z_norm, i_norm)
ax3.set_xlim(iz_xlim)
ax3.set_ylim(iz_ylim)
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(z_norm, g_norm)
ax4.set_xlim(gz_xlim)
ax4.set_ylim(gz_ylim)
ax4.set_title("Normalized G vs. Normalized Z")
plt.savefig("Generated Noisy Fluxes.png")



# Apply MCMC to sample the XDC probability
tot_samp = 100000 * 100
dim = xmean.shape[1]

#figure out the number of gaussins in the model
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
#plt.scatter(samples[:,0],samples[:,1],s=1,alpha=0.1)
#print(samples)

xmean_g = samples[:, 0]
xmean_r = samples[:, 1]
xmean_i = samples[:, 2]
xmean_z = samples[:, 3]

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
ax1.scatter(xmean_r, xmean_g, alpha = 0.1)
ax1.set_xlim([0.018689261934454145, 0.30570522434917147])
ax1.set_ylim([-0.02498731864164616, 0.37520237750109764])
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(xmean_i, xmean_r, alpha = 0.1)
ax2.set_xlim([0.168327331025541, 0.3818603775513799])
ax2.set_ylim([0.01591015407155072, 0.3084843322120745])
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(xmean_z, xmean_i, alpha = 0.1)
ax3.set_xlim([0.10870964061586308, 0.7513928898988901])
ax3.set_ylim([0.16554822316263784, 0.38463948541428344])
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(xmean_z, xmean_g, alpha = 0.1)
ax4.set_xlim([0.10870964061586297, 0.75139288989889])
ax4.set_ylim([-0.024987318641646213, 0.3752023775010975])
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("MCMC Samples")
plt.savefig("MCMC_Samples.png")

#plt.show()
# Find KL for the original noisy input 
q, indexes = bin_data(ndata, neurons)
#print(q)
prob = kl(apply_norm(q), p)
print(prob)
with open('Before_Deconv_kl.txt', 'w') as filehandle:
    filehandle.write(str(prob)+'\n')

# Use MCMC for the KL divergence test
q, indexes = bin_data(samples, neurons)
#print(q)

#count = 0
#q = []
#for i in range(len(count_samples)):
#    num_samples = count_samples[i]
#    s = 0
#    for j in range(count, count + num_samples):
#        diff = neurons[i] - samples[j]
#        scalar = np.matmul(np.matmul(diff.T, li.inv(V[i])), diff)
#        gaussian =(2*np.pi)**(-d/2) * li.det(V[i])**(-1/2) * np.exp(-scalar/2)
#        s += xamp1[i] * gaussian 
#    q.append(s)
#    count += num_samples

    
prob = kl(apply_norm(q), p)
print(prob)
with open('MCMC_kl_results.txt', 'w') as filehandle:
    filehandle.write(str(prob)+'\n')

# Plot the contours
V = xcovar
xamp = xamp1
old_samples = samples.copy()
g_factor = 50 / np.mean(samples[:, 0])
r_factor = 50 / np.mean(samples[:, 1])
i_factor = 50 / np.mean(samples[:, 2])
z_factor = 50 / np.mean(samples[:, 3])
samples[:, 0] *= np.repeat([g_factor], samples.shape[0], axis = 0)
samples[:, 1] *= np.repeat([r_factor], samples.shape[0], axis = 0)
samples[:, 2] *= np.repeat([i_factor], samples.shape[0], axis = 0)
samples[:, 3] *= np.repeat([z_factor], samples.shape[0], axis = 0)
#print(samples[:,0])
g_diff = int(np.amax(samples[:,0]) - np.amin(samples[:,0]))+1
print(g_diff)
r_diff = int(np.amax(samples[:,1]) - np.amin(samples[:,1]))+1
i_diff = int(np.amax(samples[:,2]) - np.amin(samples[:,2]))+1
z_diff = int(np.amax(samples[:,3]) - np.amin(samples[:,3]))+1
g_min = int(np.amin(samples[:,0]))
r_min = int(np.amin(samples[:,1]))
i_min = int(np.amin(samples[:,2]))
z_min = int(np.amin(samples[:,3]))

image_gr = np.zeros([g_diff, r_diff],dtype=np.double)
image_ri = np.zeros([r_diff, i_diff],dtype=np.double)
image_iz = np.zeros([i_diff, z_diff],dtype=np.double)
image_gz = np.zeros([g_diff, z_diff,],dtype=np.double)
d=4
#factor_arr = np.repeat([20], d, axis = 0)
count = 0
#print(count_samples)
for i in range(len(count_samples)):
    num_samples = count_samples[i]
    for j in range(count, count + num_samples):
        #diff = bin_xmean[i][j]-xmean[i]
        #print(samples[j])
        diff = old_samples[j] - xmean[i]
        scalar = np.matmul(np.matmul(diff.T, li.inv(V[i])), diff)
        #gaussian =np.power((2*np.pi),(-d/2)) * np.power(li.det(V[j]),-1/2)* np.exp((-1/2)*scalar)
        gaussian = (2*np.pi)**(-d/2) * (li.det(V[i]))**(-1/2)* np.exp((-1/2)*scalar)
        (g, r, i, z) = np.asarray(samples[j, :], np.int32)
#        print(g)
        image_gr[g - g_min, r - r_min] += xamp[i] * gaussian
        image_ri[r - r_min, i - i_min] += xamp[i] * gaussian
        image_iz[i - i_min, z - z_min] += xamp[i] * gaussian
        image_gz[g - g_min, z - z_min] += xamp[i] * gaussian
    count += num_samples
#new_q = np.where(q < np.amin(p[np.where(p != 0)]), 0, q)

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
ax1.contour(image_gr)
#ax1.set_xlim([0.018689261934454145, 0.30570522434917147])
#ax1.set_ylim([-0.02498731864164616, 0.37520237750109764])
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.contour(image_ri)
#ax2.set_xlim([0.168327331025541, 0.3818603775513799])
#ax2.set_ylim([0.01591015407155072, 0.3084843322120745])
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.contour(image_iz)
#ax3.set_xlim([0.10870964061586308, 0.7513928898988901])
#ax3.set_ylim([0.16554822316263784, 0.38463948541428344])
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.contour(image_gz)
#ax4.set_xlim([0.10870964061586297, 0.75139288989889])
#ax4.set_ylim([-0.024987318641646213, 0.3752023775010975])
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("MCMC Contours")
plt.savefig("MCMC_Contours.png")

#ax = plt.gca()
#print(indices)
#plt.contour(image_gr)
##ax.set_xlim(xlim)
##ax.set_ylim(ylim)
#name = "Contours Representing Gaussians"
#ax.set_title(name)
#plt.savefig("Contours_Representing_Gaussians.png")

g_norm = true_data[:, 0] 
r_norm = true_data[:, 1] 
i_norm = true_data[:, 2] 
z_norm = true_data[:, 3] 

fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
#fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
ax1.scatter(r_norm, g_norm, alpha = 0.1)
gr_xlim = ax1.get_xlim()
gr_ylim = ax1.get_ylim()
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(i_norm, r_norm, alpha = 0.1)
ri_xlim = ax2.get_xlim()
ri_ylim = ax2.get_ylim()
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(z_norm, i_norm, alpha = 0.1)
iz_xlim = ax3.get_xlim()
iz_ylim = ax3.get_ylim()
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(z_norm, g_norm, alpha = 0.1)
gz_xlim = ax4.get_xlim()
gz_ylim = ax4.get_ylim()
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")

name = "Contours Over Original Image"
fig.suptitle(name)
plt.savefig("Contours_Over_Original_Image.png")
        

        

#Apply PCA to plot 4D data on a 2D grid

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#std_xmeans = StandardScaler().fit_transform(xmean)
#pca = PCA(n_components=2)
#means_comp = pca.fit_transform(std_xmeans)
#comp_min = np.min(means_comp)
##print(comp_min)
#shift_arr = np.zeros(means_comp.shape)
#shift_arr[:, :] = np.abs(comp_min)*5
#means_comp = means_comp + shift_arr
##print(means_comp)
##print(comp.shape)
#plt.close()
#plt.scatter(means_comp[:, 0], means_comp[:, 1])
#name = "Projected New Neurons"
#plt.savefig(name+".png")
#
#std_neurons = StandardScaler().fit_transform(neurons)
#neu_comp = pca.fit_transform(std_neurons)
#plt.scatter(neu_comp[:, 0], neu_comp[:, 1], c="red")
#name = "Projected New and Old Neurons"
#plt.savefig(name+".png")
#
## Convert 4D errors to 2D
#var = np.zeros([V.shape[0], V.shape[1]])
#for i in range(V.shape[0]):
#    for j in range(V.shape[1]):
#        var[i][j] = V[i][j][j]
##    print(var[:][i])
#var = np.array(var)
##print(var)
#std_var = StandardScaler().fit_transform(var)
#var_comp = pca.fit_transform(std_var)
##print(var_comp)
#V = np.zeros([var_comp.shape[0], var_comp.shape[1], var_comp.shape[1]])
##print(var_comp.shape)
#for i in range(var_comp.shape[0]):
#    for j in range(var_comp.shape[1]):
#        V[i][j][j] = var_comp[i][j]
#        #print(V[:][i][i])
##print(V)
#    #print(var[:][i])
##print(var_comp)
##print(var_comp.shape)
#
#image = np.zeros([100, 70],dtype=np.double)
#for j in range(xmean.shape[0]):
#    jXcen = np.int(means_comp[j][0])
#    jYcen = np.int(means_comp[j][1])
#    jdX = np.int(var_comp[j][0])
#    jdY = np.int(var_comp[j][1])
#    window = 2
#    for x in range(jXcen-window*jdX,jXcen+window*jdX):
#        for y in range(jYcen-window*jdY,jYcen+window*jdY):           
#            #diff = [xx[x,y], yy[x,y]]-xmean[j]
#            diff = [x, y]-means_comp[j]
#            scalar = np.matmul(np.matmul(diff.T, li.inv(V[j])), diff)
#            #gaussian =np.power((2*np.pi),(-d/2)) * np.power(li.det(V[j]),-1/2)* np.exp((-1/2)*scalar)
#            gaussian = (2*np.pi)**(-d/2) * (li.det(V[j]))**(-1/2)* np.exp((-1/2)*scalar)
#            image[y,x] += xamp1[j] * gaussian

#make the difference array, which is the difference between xmean and the index
#        diff = v - xmean[j]  This line in vector format for all pixels to make it faster instead of a loop
#factor = 800
#glen = int(factor * (np.amax(data[:, 0]) + (np.amax(data[:, 0])- np.amin(data[:, 0]))/2))
#rlen = int(factor * (np.amax(data[:, 1]) + (np.amax(data[:, 1])- np.amin(data[:, 1]))/2))
#image = np.zeros([glen,rlen],dtype=np.double)
#gg, rr=np.indices([glen,rlen])
##print(image.shape)
##print(len(xamp1))
#factor_arr = np.repeat([factor], len(xamp1), axis = 0)
#xamp1 = factor_arr * xamp1
##print(factor_arr)
#factor_arr = np.repeat(factor_arr[:,  np.newaxis], d, axis = 1)
#xmean = factor_arr * xmean
#factor_arr = np.repeat(factor_arr[:, :,  np.newaxis], d, axis = 2)
#V = np.square(factor_arr) * V
##print(xmean)
#
#
## Use the first two fluxes as the x and y values. For each x, y value, sum 
## the gaussian over the other two dimentions
#for j in range(1):
#    jGcen = np.int(xmean[j][0])
#    #print(jXcen)
#    jRcen = np.int(xmean[j][1])
#    jIcen = np.int(xmean[j][2])
#    jZcen = np.int(xmean[j][3])
#    jdG = np.int(V[j][0][0])
#    jdR = np.int(V[j][1][1])
#    jdI = np.int(V[j][2][2])
#    jdZ = np.int(V[j][3][3])
#    window = 2    
#    print(window*jdG)
#    print(window*jdR)
#    print(window*jdI)
#    print(window*jdZ)
#    
#    for g in range(jGcen-window*jdG,jGcen+window*jdG):
#        for r in range(jRcen-window*jdR,jRcen+window*jdR):
#            for i in range(jIcen-window*jdI,jIcen+window*jdI):
#                for z in range(jZcen-window*jdZ,jZcen+window*jdZ):    
#                    #print('\n')
#                    diff = [gg[g,r], rr[g,r], i, z]-xmean[j]
#                    scalar = np.matmul(np.matmul(diff.T, li.inv(V[j])), diff)
#                    #gaussian =np.power((2*np.pi),(-d/2)) * np.power(li.det(V[j]),-1/2)* np.exp((-1/2)*scalar)
#                    gaussian =(2*np.pi)**(-d/2) * (li.det(V[j]))**(-1/2)* np.exp(-scalar/2)
#                    image[r,g] += xamp1[j] * gaussian
#                    #print(image[r,g])

#plt.close()
#fig = plt.figure()   
#ax = plt.gca()
#plt.contour(image)
##ax.set_xlim(xlim)
##ax.set_ylim(ylim)
#name = "Contours Representing Gaussians"
#ax.set_title(name)
#plt.savefig(name+".png")


