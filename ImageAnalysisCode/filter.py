# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import scipy.ndimage as ndimage
import threading
import numpy as np

def median_p(data, size, axis_parallel=0):
    """Applies parallelized median filtering to data in place"""
    if size[axis_parallel]!=1:
         raise Exception('size over parrallelized axis must be 1!')
    size_here = [size[i] for i in range(len(size)) if i != axis_parallel]
    for z in range(data.shape[axis_parallel]):
        slc = tuple([slice(None)]*axis_parallel+[z])
        id = threading.Thread(target=ndimage.median_filter, args=(data[slc],), kwargs={'size':size_here, 'output':data[slc]})
        id.start()
    id.join()

def gaussian_p(data, sigma, axis_parallel=0):
    """Applies parallelized median filtering to data in place"""
    if sigma[axis_parallel]!=0:
         raise Exception('sigma over parrallelized axis must be 0!')
    sigma_here = [sigma[i] for i in range(len(sigma)) if i != axis_parallel]
    for z in range(data.shape[axis_parallel]):
        slc = tuple([slice(None)]*axis_parallel+[z])
        id = threading.Thread(target=ndimage.gaussian_filter, args=(data[slc],), kwargs={'sigma':sigma_here, 'output':data[slc]})
        id.start()
    id.join()
    

def low_mean(data, axis, stop, median=False):
    """Takes the mean over lowes up_to (if int) or up_to*len (if float) entrys along axis"""
    slc = tuple([slice(None)]*axis + [slice(0,stop if type(stop) is int else round(stop*data.shape[axis]))])
    if median:
        return np.median(np.sort(data, axis=axis)[slc], axis=axis)
    else:
        return np.mean(np.sort(data, axis=axis)[slc], axis=axis)
def high_mean(data, axis, start, median=False):
    """Takes the mean over lowes up_to (if int) or up_to*len (if float) entrys along axis"""
    slc = tuple([slice(None)]*axis + [slice(start if type(start) is int else round(start*data.shape[axis]),data.shape[axis])])
    if median:
        return np.median(np.sort(data, axis=axis)[slc], axis=axis)
    else:
        return np.mean(np.sort(data, axis=axis)[slc], axis=axis)

def linear_fit(points):
    xs = np.arange(len(points))
    poly = np.polyfit(xs, points, 1)
    return poly[1]+poly[0]*xs

def correct_black_and_int(data, blackfrac, intfrac, axis):
    """
    Corrects the black lvl and intensity of the data along the given axis."""
    if axis==0:
        blacks_t = low_mean(data.reshape((data.shape[0],-1)), 1, blackfrac)
        blacks_t = np.round(np.clip(linear_fit(blacks_t), 0, np.inf)).astype(np.int32)
        data = np.clip(data-blacks_t[:, np.newaxis, np.newaxis, np.newaxis],0,np.inf).astype(np.uint16)
        intensities_t = high_mean(data.reshape((data.shape[0],-1)), 1, intfrac)
        intensities_t = linear_fit(intensities_t)
        intensities_t = intensities_t/intensities_t.mean()
        data = np.round(data/intensities_t[:, np.newaxis, np.newaxis, np.newaxis]).astype(np.uint16)
        return data
    elif axis==1:
        blacks_z = low_mean(np.moveaxis(data,0,1).reshape((data.shape[1],-1)), 1, blackfrac)
        blacks_z = np.round(np.clip(linear_fit(blacks_z), 0, np.inf)).astype(np.int32)
        data = np.clip(data-blacks_z[np.newaxis, :, np.newaxis, np.newaxis],0,np.inf).astype(np.uint16)
        intensities_z = high_mean(np.moveaxis(data,0,1).reshape((data.shape[1],-1)), 1, intfrac)
        intensities_z = linear_fit(intensities_z)
        intensities_z = intensities_z/intensities_z.mean()
        data = np.round(data/intensities_z[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.uint16)
        return data
    else:
        raise Exception('Only implemented for axis=1,2')
