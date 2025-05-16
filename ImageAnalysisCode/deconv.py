# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

# This is code for automated blind deconvolution of 3D and 2D images using JAX.

import numpy as np
from skimage.restoration import richardson_lucy
from scipy import ndimage
import jax
import jax.numpy as jnp
from  jax.scipy.signal import fftconvolve as fftconvolve, convolve


def find_psf_centers(data, hist_threshold, N_testpoints):
    """Find bright pixels as possible sample PSF centers."""
    hist, bins = np.histogram(data, bins=1000, density=True)
    cumhist = np.cumsum(hist)
    cumhist = cumhist/cumhist[-1]
    maxval = bins[np.argmax(cumhist>hist_threshold)]

    print("Pixelsvalues above {:.1f} taken, of which in dataset are {}, meaning fraction of {:.1e}.".format(
                    maxval, np.sum(data>maxval), np.sum(data>maxval)/data.size))
    maxincs = np.where(data>maxval)
    to_take = np.random.permutation(len(maxincs[0]))[:N_testpoints]
    maxincs_sel = [l[to_take] for l in maxincs]
    return maxincs_sel


def build_psf_testset(data, psf_size, psf_center_idcs):
    """Gathers examples to estimate PSF. psf_size ordered as data"""
    psf_z, psf_x, psf_y = psf_size
    max_allowed = np.iinfo(data.dtype).max
    testset = np.empty((len(psf_center_idcs[0]),psf_z, psf_x, psf_y), dtype=data.dtype)
    testset.fill(max_allowed)
    im_padded = np.pad(data, ((0,0), (psf_z//2,psf_z//2), (psf_x//2,psf_x//2), (psf_y//2,psf_y//2)), 
                        mode='constant', constant_values=np.array(max_allowed).astype(np.uint16))
    for i in range(len(psf_center_idcs[0])):
        temp = im_padded[psf_center_idcs[0][i],
                        psf_center_idcs[1][i]:psf_center_idcs[1][i]+psf_z, 
                        psf_center_idcs[2][i]:psf_center_idcs[2][i]+psf_x,
                        psf_center_idcs[3][i]:psf_center_idcs[3][i]+psf_y].copy().astype(np.float32)
        valid = temp<max_allowed
        temp[valid] = np.clip(np.round((max_allowed-1)*(temp[valid]/temp[psf_z//2,psf_x//2,psf_y//2])), 0, max_allowed-1).astype(np.uint16)
        testset[i] = temp
    testset.sort(axis=0)
    return testset

def gather_psf_from_testset(testset, psf_size, frac_to_average, median):
    """Gathers sample PSFs (i.e. illumination around bright pixels) from testset"""
    max_allowed = np.iinfo(testset.dtype).max
    psf_z, psf_x, psf_y = psf_size
    psf = np.zeros(psf_size)
    ns_samples=[]
    average = np.median if median else np.mean
    for z in range(psf_z):
        for x in range(psf_x):
            for y in range(psf_y):
                n_samples=np.argmax(testset[:,z,x,y]==max_allowed)
                if n_samples==0 and not testset[0,z,x,y]==max_allowed:
                    n_samples=len(testset)
                ns_samples.append(n_samples)
                psf[z,x,y] = average(testset[:round(n_samples*frac_to_average),z,x,y])
    print("smallest_number of samples availabel: ", np.array(ns_samples).min())
    return psf


def estimate_psf(data, hist_threshold, N_testpoints, psf_size, frac_to_average, symmetrize = True, normalize=True, zero_zbound=True, median=False):
    """
    Estimates the PSF from data.
        Parameters:
            data: Data as t x z x x x y stack
            hist_threshold: values above this cumultative density histogram threshhold are
                considered possible center pixel
            N_testpoints: Number of centerpixels to be chosen randomly from available
            psf_size: size of the PSF, z x x x y
            frac_to_average: which fraction of found pixel to use (smallest values are averaged)
    """
    psf_center_idcs = find_psf_centers(data, hist_threshold, N_testpoints)
    testset = build_psf_testset(data, psf_size, psf_center_idcs)
    psf = gather_psf_from_testset(testset, psf_size, frac_to_average, median=median)
    if symmetrize:
        psf = (psf + np.flip(psf, axis=1) + np.flip(psf, axis=2) + np.flip(psf, axis=(1,2)))/4
    if zero_zbound:
        psf[0] = 0
        psf[-1] = 0
    if normalize:
        psf = psf/psf.sum()
    return psf

def compute_reduced_psf(psf):
    """Computes reduced PSF for padding"""
    reduced_psf = np.array([richardson_lucy(psf[i], psf[len(psf)//2]) for i in range(len(psf))])
    reduced_psf = reduced_psf/(reduced_psf[len(psf)//2].sum())
    return reduced_psf

def pad_by_psf(data, psf=None, reduced_psf=None, factor_padding=1, make_z_even=True):
    """Pads data by the PSF, so that the convolution with the PSF is valid."""
    if reduced_psf is None:
        reduced_psf = compute_reduced_psf(psf)
    padfront = np.array([[np.round(ndimage.convolve(f[0], reduced_psf[i])).astype(data.dtype) for i in range(len(reduced_psf)//2)] for f in data], dtype=data.dtype)
    padback = np.array([[np.round(ndimage.convolve(f[-1], reduced_psf[len(reduced_psf)//2+i])).astype(data.dtype) for i in range(len(reduced_psf)//2)] for f in data], dtype=data.dtype)
    padded = np.concatenate((factor_padding*padfront, data, factor_padding*padback), axis=1)
    offset = padfront.shape[1]
    if make_z_even is True and padded.shape[1]%2==1:
        padded = np.concatenate((np.zeros_like(padded[:,0:1]), padded), axis=1)
        offset+=1
    return padded, (slice(None), slice(offset, offset+data.shape[1]))

lap_kernel = ndimage.generate_binary_structure(3,1).astype(np.float32)
lap_kernel[1,1,1]=-6

def loss(image_n, psf_n, image_i, norm_loss=.3, norm_loss_psf=1e4, reg_loss=.3, reg_loss_psf=5e6, lap_loss_f=1e-3, lap_loss_psf_f=5e6, reg_resize=0.5):
    """Loss function for deconvolution"""
    conv = fftconvolve(image_n, psf_n, mode='same')
    im_loss = jnp.sum(jnp.where(image_i==-1, 0, (image_i-conv)**2))
    normalization_loss = norm_loss*jnp.abs(jnp.sum(image_n)-jnp.sum(image_i))
    normalizing_psf = norm_loss_psf*jnp.abs(jnp.sum(psf_n)-1)
    regularization_loss = reg_loss*jnp.sum(jnp.arctan(image_n/reg_resize)) + reg_loss_psf*jnp.sum(psf_n**4)
    lap_loss = lap_loss_f*jnp.sum(fftconvolve(image_n, lap_kernel, mode='same')**2)
    lap_loss_psf = lap_loss_psf_f*jnp.sum(convolve(psf_n, lap_kernel, mode='same', method='direct')**2)
    return im_loss + normalization_loss + normalizing_psf + regularization_loss + lap_loss + lap_loss_psf

def loss_parallel(images_n, psf_n, images_i, **kwargs):
    """Parallelized loss function for deconvolution"""
    return jnp.sum(jax.vmap(lambda a,b,c: loss(a,b,c, **kwargs), in_axes=(0, None, 0))(images_n, psf_n, images_i))


def deconvolve_blind(data, psf, batchsize=2, epochs=500, factor_padding=0.5, kwargs_loss={}, print_every=50):
    """Blind deconvolution of data using the given PSF as initial guess."""
    images_i, sliceback = pad_by_psf(data.astype(np.float32), psf=psf.astype(np.float32), factor_padding=factor_padding)

    psf_i = psf.copy().astype(np.float32)
    #image_i = image_i/(image_i).sum()
    #images_i = images_i/(images_i).sum(axis=(1,2,3))[:,np.newaxis, np.newaxis, np.newaxis]
    images_n = images_i.copy()
    psf_n=psf_i.copy()

    grads_psf = np.zeros_like(psf_n)
    ramp_in = 100
    ramp_out = epochs-250

    if not 'reg_resize' in kwargs_loss.keys():
        kwargs_loss['reg_resize'] = 1#np.median(np.sort(images_i.flat)[-images_i.size//100000:])/5

    loss_and_grads = jax.jit(jax.value_and_grad(lambda a,b,c: loss_parallel(a,b,c, **kwargs_loss), argnums=(0,1)))
    lossval=0

    def lr_im(i):
        lr=1e-1#1.
        if i<ramp_in: return (i+1)/ramp_in*lr
        elif i >=ramp_in and i <= ramp_out: return lr
        elif i > ramp_out: return lr*0.5**((i-ramp_out)/50)

    def lr_psf(i):
        lr=1e-9#50.
        if i<ramp_in: return (i+1)/ramp_in*lr
        elif i >=ramp_in and i <=ramp_out: return lr
        elif i > ramp_out: return lr*0.5**((i-ramp_out)/50)

    for i in range(epochs):
        k=0
        lossval = 0
        while k < len(images_i):
            to = np.minimum(k+batchsize, len(images_i))
            lossval_temp, grads_temp = loss_and_grads(images_n[k:to], psf_n, images_i[k:to])
            lossval += lossval_temp
            images_n[k:to] = np.clip(images_n[k:to] - lr_im(i)*grads_temp[0], 0, np.inf)
            grads_psf += grads_temp[1]/len(images_i)
            k += batchsize
        psf_n = np.clip(psf_n - lr_psf(i)*grads_psf, 0, np.inf)
        grads_psf = np.zeros_like(psf_n)

        if not print_every is None and (i%print_every==0 or i==(epochs-1)):
            print("Epoch {:03d}, Loss: {:.3e}, Normalization Image: {:.5f}, Normalization PSF: {:.5f}".format(
                i, lossval, images_n.sum(), psf_n.sum()))
    grads_temp=None
    images_n = images_n[sliceback]
    multfact = 1/np.max(images_n)*(np.iinfo(np.uint16).max-10)
    return np.round(images_n*multfact).astype(np.uint16), psf_n, multfact


lap_kernel2D = ndimage.generate_binary_structure(2,1).astype(np.float32)
lap_kernel2D[1,1]=-4

def loss2D(image_n, psf_n, image_i, norm_loss=.3, norm_loss_psf=1e4, reg_loss=.3, reg_loss_psf=5e6, lap_loss_f=1e-3, lap_loss_psf_f=5e6, reg_resize=0.5):
    """Loss function for 2D deconvolution"""
    conv = fftconvolve(image_n, psf_n, mode='same')
    im_loss = jnp.sum(jnp.where(image_i==-1, 0, (image_i-conv)**2))
    normalization_loss = norm_loss*jnp.abs(jnp.sum(image_n)-jnp.sum(image_i))
    normalizing_psf = norm_loss_psf*jnp.abs(jnp.sum(psf_n)-1)
    regularization_loss = reg_loss*jnp.sum(jnp.arctan(image_n/reg_resize)) + reg_loss_psf*jnp.sum(psf_n**4)
    lap_loss = lap_loss_f*jnp.sum(fftconvolve(image_n, lap_kernel2D, mode='same')**2)
    lap_loss_psf = lap_loss_psf_f*jnp.sum(convolve(psf_n, lap_kernel2D, mode='same', method='direct')**2)
    return im_loss + normalization_loss + normalizing_psf + regularization_loss + lap_loss + lap_loss_psf

def loss_parallel2D(images_n, psf_n, images_i, **kwargs):
    """Parallelized loss function for 2D deconvolution"""
    return jnp.sum(jax.vmap(lambda a,b,c: loss2D(a,b,c, **kwargs), in_axes=(0, None, 0))(images_n, psf_n, images_i))

def deconvolve_blind_2D(data, psf, batchsize=2, epochs=500, factor_padding=0.5, kwargs_loss={}, print_every=50):
    """Blind deconvolution of data using the given PSF as initial guess."""
    images_i = data.astype(np.float32)

    psf_i = psf.copy().astype(np.float32)
    images_n = images_i.copy()
    psf_n=psf_i.copy()

    grads_psf = np.zeros_like(psf_n)
    ramp_in = 100
    ramp_out = epochs-250

    if not 'reg_resize' in kwargs_loss.keys():
        kwargs_loss['reg_resize'] = 1

    loss_and_grads = jax.jit(jax.value_and_grad(lambda a,b,c: loss_parallel2D(a,b,c, **kwargs_loss), argnums=(0,1)))
    lossval=0

    def lr_im(i):
        lr=1e-1#1.
        if i<ramp_in: return (i+1)/ramp_in*lr
        elif i >=ramp_in and i <= ramp_out: return lr
        elif i > ramp_out: return lr*0.5**((i-ramp_out)/50)

    def lr_psf(i):
        lr=1e-9#50.
        if i<ramp_in: return (i+1)/ramp_in*lr
        elif i >=ramp_in and i <=ramp_out: return lr
        elif i > ramp_out: return lr*0.5**((i-ramp_out)/50)

    for i in range(epochs):
        k=0
        lossval = 0
        while k < len(images_i):
            to = np.minimum(k+batchsize, len(images_i))
            lossval_temp, grads_temp = loss_and_grads(images_n[k:to], psf_n, images_i[k:to])
            lossval += lossval_temp
            images_n[k:to] = np.clip(images_n[k:to] - lr_im(i)*grads_temp[0], 0, np.inf)
            grads_psf += grads_temp[1]/len(images_i)
            k += batchsize
        psf_n = np.clip(psf_n - lr_psf(i)*grads_psf, 0, np.inf)
        grads_psf = np.zeros_like(psf_n)

        if not print_every is None and (i%print_every==0 or i==(epochs-1)):
            print("Epoch {:03d}, Loss: {:.3e}, Normalization Image: {:.5f}, Normalization PSF: {:.5f}".format(
                i, lossval, images_n.sum(), psf_n.sum()))
    grads_temp=None
    multfact = 1/np.max(images_n)*(np.iinfo(np.uint16).max-10)
    return np.round(images_n*multfact).astype(np.uint16), psf_n, multfact
    

def deconvolve_background(data, psf, epochs=500, factor_padding=0.5, kwargs_loss={}, print_every=50):
    """Blind deconvolution of data using the given PSF as initial guess, for data without time axis."""
    image_n, sliceback = pad_by_psf(data[np.newaxis].astype(np.float32), psf=psf.astype(np.float32), factor_padding=factor_padding)
    image_n = image_n[0]

    image_i = - np.ones_like(image_n)
    image_i[sliceback[1:]] = image_n[sliceback[1:]]

    ramp_in = 100
    ramp_out = epochs-250

    if not 'reg_resize' in kwargs_loss.keys():
        kwargs_loss['reg_resize'] = 1#np.median(np.sort(images_i.flat)[-images_i.size//100000:])/5

    loss_and_grads = jax.jit(jax.value_and_grad(lambda a,b,c: loss(a,b,c, **kwargs_loss), argnums=0))
    lossval=0

    def lr_im(i):
        lr=1e-1#1.
        if i<ramp_in: return (i+1)/ramp_in*lr
        elif i >=ramp_in and i <= ramp_out: return lr
        elif i > ramp_out: return lr*0.5**((i-ramp_out)/50)

    for i in range(epochs):
        lossval, grads = loss_and_grads(image_n, psf, image_i)
        image_n = np.clip(image_n - lr_im(i)*grads, 0, np.inf)

        if not print_every is None and (i%print_every==0 or i==(epochs-1)):
            print("Epoch {:03d}, Loss: {:.3e}, Normalization Image: {:.5f}, Normalization PSF: {:.5f}".format(
                i, lossval, image_n.sum(), psf.sum()))
            
    image_n = image_n[np.newaxis][sliceback][0]
    multfact = 1/np.max(image_n)*(np.iinfo(np.uint16).max-10)
    return np.round(image_n*multfact).astype(np.uint16), multfact