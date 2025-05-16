# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
from scipy import ndimage
import jax
import jax.numpy as jnp

def jax_crosscor(img1, img2, Dx, Dy):
    """Compute cross-correlation between two images with given displacements"""
    Lx, Ly = img1.shape[0], img1.shape[1]
    xgrid, ygrid = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
    rollim = jnp.roll(jnp.roll(img1, Dx, axis=0), Dy, axis=1)
    prodimg = rollim*img2
    valid = ((Dx-1)<xgrid)*(xgrid<(Lx+Dx))*((Dy-1)<ygrid)*(ygrid<(Ly+Dy))
    summed = jnp.sum((prodimg*valid))/(jnp.sum(rollim)*jnp.sum(img2)*np.sum(valid)/valid.size)
    return summed

"""Vectorized cross-correlation function"""
jax_vcrosscor = jax.jit(jax.vmap(jax.vmap(jax_crosscor, in_axes=(None,None,None,0)), in_axes=(None,None,0,None)))

def shift_align(im_shifted, im_base, shifts = np.arange(-13,13)):
    """maximizes the cross-correlation between two images by shifting the first one"""
    cc_matrix = jax_vcrosscor((im_shifted-jnp.min(im_shifted)).astype(float), (im_base-jnp.min(im_base)).astype(float), shifts, shifts)
    maxpos = np.unravel_index(np.array(cc_matrix).argmax(), cc_matrix.shape)
    return np.array((shifts[maxpos[0]], shifts[maxpos[1]])), cc_matrix # im 1 should be shifted like that


def build_shiftlist_from_image(im, shiftrange = np.arange(-13,13), sigma=2, mindist_to_take=2, maxdist_to_take=10, return_shiftlist=False):
    """Moving through a stack im, this function builds a shiftlist for each image in the stack.
    The shiftlist contains the shifts that are needed to align the images in the stack.
    The function uses the cross-correlation method to find the shifts."""
    shiftlist = np.zeros((len(im),len(im),2))*np.nan
    for i in range(0,len(im)):
        for j in range(0, len(im)):
            if np.abs(i-j)<mindist_to_take or np.abs(i-j)>maxdist_to_take:
                continue
            shift_new = shift_align(im[i].astype(float), im[j].astype(float), shifts=shiftrange)[0]
            shiftlist[i,j] = shift_new
    
    def loss(p, cdist):
        cdistguess = p[jnp.newaxis]-p[:,jnp.newaxis]
        return jnp.sum((cdistguess-cdist)[~np.isnan(cdist)]**2)

    grad = jax.value_and_grad(loss)

    p0_0 = np.nanmean(shiftlist[:,:,0], axis=0)*0.
    p0_1 = np.nanmean(shiftlist[:,:,1], axis=0)*0.
    lr = 1e-3
    for i in range(1000):
        lossval0, gradval0 = grad(p0_0, shiftlist[:,:,0])
        p0_0 -= lr*gradval0
        lossval1, gradval1 = grad(p0_1, shiftlist[:,:,1])
        p0_1 -= lr*gradval1
    #print(lossval0, lossval1)
    shiftlistFit = -np.round(ndimage.gaussian_filter(np.stack((p0_0, p0_1), axis=-1), (sigma,0), mode='nearest')).astype(int)
    if return_shiftlist:
        return shiftlistFit, shiftlist
    return shiftlistFit

def shift_image(im, shiftlistFit):
    """Shifts the image im according to the shiftlistFit"""
    shifted = np.empty_like(im).astype(float)
    shifted.fill(np.nan)
    Lx, Ly = shifted.shape[1], shifted.shape[2]
    for i in range(0,len(im)):
        xgrid, ygrid = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
        rollim = np.roll(np.roll(im[i], shiftlistFit[i][0], axis=0), shiftlistFit[i][1], axis=1)
        valid = ((shiftlistFit[i][0]-1)<xgrid)*(xgrid<(Lx+shiftlistFit[i][0]))*((shiftlistFit[i][1]-1)<ygrid)*(ygrid<(Ly+shiftlistFit[i][1]))
        shifted[i][valid] = rollim[valid]
    return shifted

def shift_series(series, shiftlistFit, nanval = 0):
    return np.array([np.nan_to_num(shift_image(im, shiftlistFit), nan=nanval).astype(im.dtype) for im in series])