#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:01:39 2019

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
import matplotlib.colors as clr

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    print(cols.info)
    first_two_rows = data[:2]
#    print(first_two_rows)
print(header)

os.chdir(os.getcwd()+"/xdc_t4")

neurons = []
with open('neurons.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        neurons.append(a.astype(np.float))
neurons = np.array(neurons)

xmean = []
with open('new_xmeans.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        xmean.append(a.astype(np.float))
xmean = np.array(xmean)

xamp = []
with open('new_xamps.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        val = float(l)
        xamp.append(val)
xamp = np.array(xamp)
#print(xamp)

os.chdir("..")

if os.path.isdir("xdc_plot_t4"):
    shutil.rmtree("xdc_plot_t4")
os.mkdir("xdc_plot_t4")
os.chdir(os.getcwd()+"/xdc_plot_t4")

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
#fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
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
fig.suptitle("True Fluxes")
plt.savefig("True_Fluxes.png")
#plt.show()

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
#fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
ax1.scatter(r_norm / i_norm, g_norm / r_norm, alpha = 0.1)
r1_xlim = ax1.get_xlim()
r1_ylim = ax1.get_ylim()
ax1.set_xlabel("R/I")
ax1.set_ylabel("G/R")
ax1.set_title("G/R vs. R/I")

ax2.scatter(i_norm / z_norm, r_norm / i_norm, alpha = 0.1)
r2_xlim = ax2.get_xlim()
r2_ylim = ax2.get_ylim()
ax2.set_xlabel("I/Z")
ax2.set_ylabel("R/I")
ax2.set_title("R/I vs. I/Z")

ax3.scatter(i_norm / z_norm, g_norm / i_norm, alpha = 0.1)
r3_xlim = ax3.get_xlim()
r3_ylim = ax3.get_ylim()
ax3.set_xlabel("I/Z")
ax3.set_ylabel("G/I")
ax3.set_title("G/I vs. I/Z")

ax4.scatter(r_norm / z_norm, g_norm / i_norm, alpha = 0.1)
r4_xlim = ax4.get_xlim()
r4_ylim = ax4.get_ylim()
ax4.set_xlabel("R/Z")
ax4.set_ylabel("G/I")
ax4.set_title("G/I vs. R/Z")
fig.suptitle("Ratio of True Fluxes")
plt.savefig("Ratio_of_True_Fluxes.png")

# Read in the noisy data and their errors
ndata = np.zeros((num_rows, num_cols))
errors = np.zeros((num_rows, num_cols))
for i in range(num_cols):
    ndata[:, i] = data.field(i + 4)
    errors[:, i] = data.field(i + 8)
#print(ndata)

# Not normalized errors
nn_errors = errors.copy()   
# Normalize the noisy data and errors by the same value
for i in range(num_rows):
    total_flux = sum(ndata[i, :])
    ndata[i, :] = ndata[i, :] / total_flux
    errors[i, :] = errors[i, :] / total_flux
#print(ndata)
    
#plt.close()
#g_hist, g_edges = np.histogram(errors[:, 0], bins = 40, range = [-.25, 1.25])
#print(g_edges)
#plt.xticks(g_edges, rotation = 'vertical')
#plt.hist(g_hist, g_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Errors in G")
#
#plt.close()
#r_hist, r_edges = np.histogram(errors[:, 1], bins = 40, range = [-.25, 1.25])
#print(r_edges)
#plt.xticks(r_edges, rotation = 'vertical')
#plt.hist(r_hist, r_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Errors in R")
#
#plt.close()
#i_hist, i_edges = np.histogram(errors[:, 2], bins = 40, range = [-.25, 1.25])
#print(i_edges)
#plt.xticks(i_edges, rotation = 'vertical')
#plt.hist(i_hist, i_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Errors in I")
#
#plt.close()
#z_hist, z_edges = np.histogram(errors[:, 3], bins = 40, range = [-.25, 1.25])
#print(z_edges)
#plt.xticks(z_edges, rotation = 'vertical')
#plt.hist(z_hist, z_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Errors in Z")

#Plot errors that aren't normalized
#plt.close()
#g_hist, g_edges = np.histogram(nn_errors[:, 0], bins = 40)
#print(g_edges)
#plt.xticks(g_edges, rotation = 'vertical')
#plt.hist(g_hist, g_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Non-Normalized Errors in G")
#
#plt.close()
#r_hist, r_edges = np.histogram(nn_errors[:, 1], bins = 40)
#print(r_edges)
#plt.xticks(r_edges, rotation = 'vertical')
#plt.hist(r_hist, r_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Non-Normalized Errors in R")
#
#plt.close()
#i_hist, i_edges = np.histogram(nn_errors[:, 2], bins = 40)
#print(i_edges)
#plt.xticks(i_edges, rotation = 'vertical')
#plt.hist(i_hist, i_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Non-Normalized Errors in I")
#
#plt.close()
#z_hist, z_edges = np.histogram(nn_errors[:, 3], bins = 40)
#print(z_edges)
#plt.xticks(z_edges, rotation = 'vertical')
#plt.hist(z_hist, z_edges)
#plt.tight_layout()
#plt.savefig("Histogram of Non-Normalized Errors in Z")
#
## Plot normalized errors over normalized values
#plt.close()
#fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
#fig.tight_layout()
#ax1.scatter(g_norm, errors[:, 0])
#ax1.set_ylim([-.25, 0.5])
#ax1.set_title("Normalized Error vs. True G")
#ax2.scatter(i_norm, errors[:, 1])
#ax2.set_ylim([-.25, 0.5])
#ax2.set_title("Normalized Error vs. True R")
#ax3.scatter(r_norm, errors[:, 2])
#ax3.set_ylim([-.25, 0.75])
#ax3.set_title("Normalized Error vs. True I")
#ax4.scatter(z_norm, errors[:, 3])
#ax4.set_ylim([-.25, 1])
#ax4.set_title("Normalized Error vs. True Z")
#plt.savefig("Errors on Normalized True Fluxes.png")


g_norm = ndata[:, 0] 
r_norm = ndata[:, 1] 
i_norm = ndata[:, 2] 
z_norm = ndata[:, 3] 

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
ax1.scatter(r_norm, g_norm, alpha = 0.1)
ax1.set_xlim([-1, 1.5])
ax1.set_ylim([-0.50, 1])
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(i_norm, r_norm, alpha = 0.1)
ax2.set_xlim([-0.2, 1.5])
ax2.set_ylim([-0.3, 0.75])
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(z_norm, i_norm, alpha = 0.1)
ax3.set_xlim([-1, 1.25])
ax3.set_ylim([-0.5, 1.2])
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(z_norm, g_norm, alpha = 0.1)
ax4.set_xlim([-1, 1.5])
ax4.set_ylim([-0.5, 0.75])
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("Noisy Fluxes")
plt.savefig("Noisy_Fluxes.png")

print(gr_xlim)
print(gr_ylim)

print(ri_xlim)
print(ri_ylim)

print(iz_xlim)
print(iz_ylim)

print(gz_xlim)
print(gz_ylim)


# Plot a histogram for the errors of each flux
#plt.close()
#fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
#fig.tight_layout()
#ax1.scatter(g_norm, errors[:, 0])
#ax1.set_title("Normalized Error vs. Noisy G")
#ax2.scatter(i_norm, errors[:, 1])
#ax2.set_title("Normalized Error vs. Noisy R")
#ax3.scatter(r_norm, errors[:, 2])
#ax3.set_title("Normalized Error vs. Noisy I")
#ax4.scatter(z_norm, errors[:, 3])
#ax4.set_title("Normalized Error vs. Noisy Z")
#plt.savefig("Errors on Normalized Noisy Fluxes.png")

neurons_g = neurons[:, 0]
neurons_r = neurons[:, 1]
neurons_i = neurons[:, 2]
neurons_z = neurons[:, 3]

xmean_g = xmean[:, 0]
xmean_r = xmean[:, 1]
xmean_i = xmean[:, 2]
xmean_z = xmean[:, 3]

# Plot the original neuron's g and r fluxes
plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
ax1.scatter(neurons_r, neurons_g, alpha = 0.1)
#ax1.set_xlim(gr_xlim)
#ax1.set_ylim(gr_ylim)
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(neurons_i, neurons_r, alpha = 0.1)
#ax2.set_xlim(ri_xlim)
#ax2.set_ylim(ri_ylim)
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(neurons_z, neurons_i, alpha = 0.1)
#ax3.set_xlim(iz_xlim)
#ax3.set_ylim(iz_ylim)
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(neurons_z, neurons_g, alpha = 0.1)
#ax4.set_xlim(gz_xlim)
#ax4.set_ylim(gz_ylim)
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("Neuron Fluxes Before Deconvolution")
plt.savefig("Neuron_Fluxes_Before_Deconvolution.png")

#print(factor)
plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
factor = 1/np.max(xamp)
print(factor)
#rgba_colors = np.zeros((len(xamp),4))
#alphas = xamp * factor
#r, g, b, a = clr.to_rgba('tab:blue')
#rgba_colors[:, 0] = r
#rgba_colors[:, 1] = g
#rgba_colors[:, 2] = b
#rgba_colors[:, 3] = alphas
ax1.scatter(xmean_r, xmean_g, alpha = 0.1)
ax1.set_xlim(gr_xlim)
ax1.set_ylim(gr_ylim)
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_title("Normalized G vs. Normalized R")

ax2.scatter(xmean_i, xmean_r, alpha = 0.1)
ax2.set_xlim(ri_xlim)
ax2.set_ylim(ri_ylim)
ax2.set_xlabel("I")
ax2.set_ylabel("R")
ax2.set_title("Normalized R vs. Normalized I")

ax3.scatter(xmean_z, xmean_i, alpha = 0.1)
ax3.set_xlim(iz_xlim)
ax3.set_ylim(iz_ylim)
ax3.set_xlabel("Z")
ax3.set_ylabel("I")
ax3.set_title("Normalized I vs. Normalized Z")

ax4.scatter(xmean_z, xmean_g, alpha = 0.1)
ax4.set_xlim(gz_xlim)
ax4.set_ylim(gz_ylim)
ax4.set_xlabel("Z")
ax4.set_ylabel("G")
ax4.set_title("Normalized G vs. Normalized Z")
fig.suptitle("Neuron Fluxes After Deconvolution")
plt.savefig("Neuron_Fluxes_After_Deconvolution.png")

plt.close()
fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2,2)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#fig.tight_layout()
ax1.scatter(xmean_r / xmean_i, xmean_g / xmean_r, alpha = 0.1)
ax1.set_xlim(r1_xlim)
ax1.set_ylim(r1_ylim)
ax1.set_xlabel("R/I")
ax1.set_ylabel("G/R")
ax1.set_title("G/R vs. R/I")

ax2.scatter(xmean_i / xmean_z, xmean_r / xmean_i, alpha = 0.1)
ax2.set_xlim(r2_xlim)
ax2.set_ylim(r2_ylim)
ax2.set_xlabel("I/Z")
ax2.set_ylabel("R/I")
ax2.set_title("R/I vs. I/Z")

ax3.scatter(xmean_i / xmean_z, xmean_g / xmean_i, alpha = 0.1)
ax3.set_xlim(r3_xlim)
ax3.set_ylim(r3_ylim)
ax3.set_xlabel("I/Z")
ax3.set_ylabel("G/I")
ax3.set_title("G/I vs. I/Z")

ax4.scatter(xmean_r / xmean_z, xmean_g / xmean_i, alpha = 0.1)
ax4.set_xlim(r1_xlim)
ax4.set_ylim(r1_ylim)
ax4.set_xlabel("R/Z")
ax4.set_ylabel("G/I")
ax4.set_title("G/I vs. R/Z")
fig.suptitle("Ratio of Fluxes After Deconvolution")
plt.savefig("Ratio_of_Fluxes_After_Deconvolution.png")


print(np.mean(errors[:,0]))
print(np.mean(errors[:,1]))
print(np.mean(errors[:,2]))
print(np.mean(errors[:,3]))
