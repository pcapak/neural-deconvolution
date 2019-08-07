#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:42:49 2019

@author: MeganT
"""

import numpy  as np
import matplotlib.pyplot as plt
import brewer2mpl
from astropy.io import fits
from scipy.stats import gaussian_kde
import os
import shutil

with fits.open(os.getcwd()+'/simulated_buzzard_data.fits') as hdul:
    hdul.info()
    data = hdul[1].data
    header = hdul[1].header
    cols = hdul[1].columns
    print(cols.info)
    first_two_rows = data[:2]

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

num_cols = 4
num_rows = data.field(0).shape[1]
true_data = []
with open('true_data.txt', 'r') as filehandle:
    lines = filehandle.readlines()
    for l in lines:
        strings = l.split(' ')
        a = np.asarray(strings[:len(strings)-1])
        true_data.append(a.astype(np.float))
true_data = np.array(true_data)

ndata = np.zeros((num_rows, num_cols))
errors = np.zeros((num_rows, num_cols))
for i in range(num_cols):
    ndata[:, i] = data.field(i + 4)
    errors[:, i] = data.field(i + 8)

os.chdir("..")

if os.path.isdir("xdc_density_t4"):
    shutil.rmtree("xdc_density_t4")
os.mkdir("xdc_density_t4")
os.chdir(os.getcwd()+"/xdc_density_t4")

bmap = brewer2mpl.get_map('Greys','sequential',9)
greys = bmap.mpl_colormap

bmap2 = brewer2mpl.get_map('Reds','sequential',9)
reds = bmap2.mpl_colormap

g_norm = true_data[:, 0] 
r_norm = true_data[:, 1] 
i_norm = true_data[:, 2] 
z_norm = true_data[:, 3] 

fig, [(grFig, riFig), (izFig, gzFig)] = plt.subplots(2, 2)
#fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
x = r_norm
y = g_norm
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
grFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
gr_xlim = grFig.get_xlim()
gr_ylim = grFig.get_ylim()
grFig.set_xlabel("R")
grFig.set_ylabel("G")
grFig.set_title("Normalized G vs. Normalized R")


#R vs I
x = i_norm
y = r_norm
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
riFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
ri_xlim = riFig.get_xlim()
ri_ylim = riFig.get_ylim()
riFig.set_xlabel("I")
riFig.set_ylabel("R")
riFig.set_title("Normalized R vs. Normalized I")

#I vs Z
x = z_norm
y = i_norm
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
izFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
iz_xlim = izFig.get_xlim()
iz_ylim = izFig.get_ylim()
izFig.set_xlabel("Z")
izFig.set_ylabel("I")
izFig.set_title("Normalized I vs. Normalized Z")

#G vs Z

x = z_norm
y = g_norm
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
gzFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
gz_xlim = gzFig.get_xlim()
gz_ylim = gzFig.get_ylim()
gzFig.set_xlabel("Z")
gzFig.set_ylabel("G")
gzFig.set_title("Normalized G vs. Normalized Z")
fig.suptitle("Density Plots of True Fluxes")
plt.savefig('true_density_plots.png')

# Plot error density 
#g_error = errors[:, 0] 
#r_error = errors[:, 1] 
#i_error = errors[:, 2] 
#z_error = errors[:, 3] 
#
#x = r_norm + r_error
#y = g_norm + g_error
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#idx = z.argsort()
#x,y,z = x[idx],y[idx],z[idx]
#grFig.scatter(x,y,c=reds(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
#grFig.set_xlim(gr_xlim)
#grFig.set_ylim(gr_ylim)
#grFig.set_title("Normalized G vs. Normalized R")
#
##R vs I
#x = i_norm + i_error
#y = r_norm + r_error
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#idx = z.argsort()
#x,y,z = x[idx],y[idx],z[idx]
#riFig.scatter(x,y,c=reds(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
#riFig.set_xlim(ri_xlim)
#riFig.set_ylim(ri_ylim)
#riFig.set_title("Normalized R vs. Normalized I")
#
##I vs Z
#x = z_norm + z_error
#y = i_norm + i_error
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#idx = z.argsort()
#x,y,z = x[idx],y[idx],z[idx]
#izFig.scatter(x,y,c=reds(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
#izFig.set_xlim(iz_xlim)
#izFig.set_ylim(iz_ylim)
#izFig.set_title("Normalized I vs. Normalized Z")
#
##G vs Z
#
#x = z_norm + z_error
#y = g_norm + g_error
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#idx = z.argsort()
#x,y,z = x[idx],y[idx],z[idx]
#gzFig.scatter(x,y,c=reds(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
#gzFig.set_xlim(gz_xlim)
#gzFig.set_ylim(gz_ylim)
#gzFig.set_title("Normalized G vs. Normalized Z")
#
#plt.savefig('true_density_with_error.png')


plt.close()

g_new = xmean[:, 0]
r_new = xmean[:, 1]
i_new = xmean[:, 2]
z_new = xmean[:, 3]

fig, [(grFig, riFig), (izFig, gzFig)] = plt.subplots(2, 2)
#fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
#G vs R
x = r_new
y = g_new
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
grFig.set_xlim(gr_xlim)
grFig.set_ylim(gr_ylim)
grFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
grFig.set_xlabel("R")
grFig.set_ylabel("G")
grFig.set_title("Normalized G vs. Normalized R")

#R vs I
x = i_new
y = r_new
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
riFig.set_xlim(ri_xlim)
riFig.set_ylim(ri_ylim)
riFig.set_xlabel("I")
riFig.set_ylabel("R")
riFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
riFig.set_title("Normalized R vs. Normalized I")

#I vs Z
x = z_new
y = i_new
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
izFig.set_xlim(iz_xlim)
izFig.set_ylim(iz_ylim)
izFig.set_xlabel("Z")
izFig.set_ylabel("I")
izFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
izFig.set_title("Normalized I vs. Normalized Z")

#G vs Z

x = z_new
y = g_new
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x,y,z = x[idx],y[idx],z[idx]
gzFig.set_xlim(gz_xlim)
gzFig.set_ylim(gz_ylim)
gzFig.set_xlabel("Z")
gzFig.set_ylabel("G")
gzFig.scatter(x,y,c=greys(z/(z[len(z)-1]*2)+0.5),s=1,edgecolor='')
gzFig.set_title("Normalized G vs. Normalized Z")

plt.savefig('xdc_density_plots.png')
#plt.show()
