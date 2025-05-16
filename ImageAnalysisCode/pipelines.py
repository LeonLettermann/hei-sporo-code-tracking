# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import tifffile
from ImageAnalysisCode.general import load_bioformat
from ImageAnalysisCode.filter import median_p, gaussian_p, correct_black_and_int, low_mean, high_mean
from ImageAnalysisCode.deconv import estimate_psf, deconvolve_blind
from ImageAnalysisCode.track import thresh_n_label,track
from ImageAnalysisCode.trajanalysis import fit_all_traj
from ImageAnalysisCode.shiftcorrect import build_shiftlist_from_image, shift_image, shift_series
from scipy import ndimage
import threading
import os


def deconvolute(data_path, series, out_path, epochs=500, returning=False, batchsize=2, shift=True, channel = 0):
    """Deconvolutes the data in data_path and saves the results in out_path.
    data_path: str, path to the data
    series: int, series number of the data
    out_path: str, path to save the results
    epochs: int, number of epochs for the deconvolution
    returning: bool, if True, the deconvoluted data is returned
    batchsize: int, batch size for the deconvolution
    shift: bool, if True, the data is realigned to compensate for drift
    channel: int, channel number of the data"""
    name = 'PC'#'Ser_{:01d}'.format(series)
    if os.path.isfile(out_path+'FiltParams.txt'):
        sporo_int = np.loadtxt(out_path+'FiltParams.txt').item()
    else:
        sporo_int = -1
    if sporo_int == -2:
        return False
    if not isinstance(data_path, str):
        data_np = data_path
    else:
        data, metadata = load_bioformat(data_path, series)
        dask = data.to_dask()[:,channel]#[::10,0]
        data_np = np.array(dask)

    #Dropping empty
    meanints = data_np.mean(axis=(1,2,3))
    dropargs = np.where(meanints<(np.median(meanints)/2))
    print('Dropping indices ', dropargs, ' for empty')
    data_np = data_np[meanints>=(np.median(meanints)/2)]

    # Preprocessing
    median_p(data_np, (1,1,3,3))
    data_np = correct_black_and_int(data_np, 0.2, 0.99999, axis=0)
    background = low_mean(data_np, 0, 0.3)
    background = ndimage.median_filter(background, (1,3,3))
    data_np = np.round(np.clip(data_np - background[np.newaxis], 0, np.inf)).astype(np.uint16)
    tifffile.imwrite(out_path + name + '_Background.tif', np.round(background/background.max()*np.iinfo(np.uint16).max).astype(np.uint16)[:,np.newaxis], imagej=True)

    if shift:
        shiftlist = build_shiftlist_from_image(background)
        background = np.nan_to_num(shift_image(background, shiftlist),nan=np.nanmin(background))
        data_np = shift_series(data_np, shiftlist)
        tifffile.imwrite(out_path + name + '_Background_shifted.tif', np.round(background/background.max()*np.iinfo(np.uint16).max).astype(np.uint16)[:,np.newaxis], imagej=True)

    background = background - background.min()
    tifffile.imwrite(out_path + name + '_Background_corr.tif', np.round(background/background.max()*np.iinfo(np.uint16).max).astype(np.uint16)[:,np.newaxis], imagej=True)
    
    data_np = correct_black_and_int(data_np, 0.2, 0.99999, axis=1)

    restremoval = data_np.copy().astype(np.float32)
    clipvals = np.sort(restremoval.reshape((len(restremoval), -1)), axis=1)[:,-restremoval[0].size//500]
    restremoval = np.clip(restremoval, 0, clipvals[:,np.newaxis, np.newaxis, np.newaxis])
    gaussian_p(restremoval, (0,0,15,15))
    median_p(restremoval, (3,1,1,1), axis_parallel=1)
    data_np = np.round(np.clip(data_np - 1.5*restremoval, 0, np.inf)).astype(np.uint16)
    restremoval = None

    tifffile.imwrite(out_path + name + '_Filtered.tif', data_np.max(axis=1)[:,np.newaxis], imagej=True)

    # Deconvolution
    psf_size = (np.minimum((data_np.shape[1]//2)*2+1, 35), 25, 25)
    psf = estimate_psf(data_np, hist_threshold = 0.9995, N_testpoints = 30000, psf_size=psf_size, frac_to_average=0.1, median=True) # 0.01
    if sporo_int==-1.: 
        sporo_int = np.median(np.sort(data_np.flat)[-data_np.size//100000:])/2
        print('Sporointensity self choosen: {}'.format(sporo_int))
    data_np = (np.clip(data_np, 0, 2*sporo_int)/(1/2*sporo_int)).astype(np.float32) 
    data_np, psf, multfact = deconvolve_blind(data_np, psf, epochs=epochs, batchsize=batchsize)


    tifffile.imwrite(out_path + name + '_Deconv.tif', data_np[:,:,np.newaxis], imagej=True)

    tifffile.imwrite(out_path + name + '_Deconv_Pr.tif', data_np.max(axis=1)[:,np.newaxis], imagej=True)
    tifffile.imwrite(out_path + name + '_PSF.tif', psf[:,np.newaxis], imagej=True)
    metadata['multfact'] = multfact
    np.save(out_path + name + '_Metadata.npy', metadata)

    data_np = np.round(data_np/data_np.max()*255).astype(np.uint8)
    if returning:
        return data_np
    
def save_boxed_sporo(sporo, labeled, original, tresh_sporos, deconv_sporos, box_positions_sporos, coms_thresh):
    """
    Saves the boxed sporo in the given path.
    sporo: int, the sporo to save
    labeled: np.ndarray, the labeled data
    original: np.ndarray, the original data
    tresh_sporos: list, the list of tresh_sporos
    deconv_sporos: list, the list of deconv_sporos
    box_positions_sporos: np.ndarray, the box positions of the sporos
    coms_thresh: np.ndarray, the coms of the tresh_sporos"""
    tresh_list, deconv_list = [], []
    for frame in range(labeled.shape[0]):
        if sporo in labeled[frame]:
            slice = ndimage.find_objects(labeled[frame]==sporo)[0]
            threshholded = labeled[frame][slice]==sporo
            tresh_list.append(threshholded)
            deconv_list.append(original[frame][slice])
            bp = np.array([slice[i].start for i in range(3)])
            box_positions_sporos[sporo,frame] = bp
            coms_thresh[sporo,frame] = ndimage.center_of_mass(threshholded)+bp
        else:
            tresh_list.append(None)
            deconv_list.append(None)
    tresh_sporos[sporo]=tresh_list
    deconv_sporos[sporo]=deconv_list



def tresh_and_track(path, base_treshold=15, dilate = 0, name = 'PC', dname='PC', max_reconn_dist = 10, min_size=20):
    """
    Thresholds and tracks the data in path and saves the results in path.
    """
    data_np = tifffile.imread(path+dname+'_Deconv.tif')
    metadata = np.load(path + dname + '_Metadata.npy', allow_pickle=True).item()
    if len(data_np.shape)==3:
        data_np = data_np[:,np.newaxis]
    data_np = ((data_np.astype(np.float32)*255)/data_np.max()).astype(np.uint8)
    data_np = thresh_n_label(data_np, ada_thresh_width = 11, ada_thresh_strength=0, abs_thresh=base_treshold, min_size=min_size, max_size=5000, min_size_per_z=0, max_size_per_z=5000, co_mask=None, dilate=dilate)
    data_np = track(data_np, metadata, max_reconn_dist, min_size=min_size)
    #tifffile.imwrite(path + name + '_Labeled.tif', data_np[:,:,np.newaxis], imagej=True)
    original = tifffile.imread(path+dname+'_Deconv.tif')
    if len(original.shape)==3:
        original = original[:,np.newaxis]
    maxlabel = data_np.max()
    tresh_sporos = [None]*(maxlabel+1)
    deconv_sporos = [None]*(maxlabel+1)
    box_positions_sporos = -np.ones(((maxlabel+1), data_np.shape[0],3), dtype=np.int16)
    coms_tresh = np.zeros(((maxlabel+1), data_np.shape[0],3), dtype=np.float16)*np.nan

    threads = []
    sbs_local = lambda i: save_boxed_sporo(i, data_np, original, tresh_sporos, deconv_sporos, box_positions_sporos, coms_tresh)
    for i in np.arange(maxlabel)+1:
        # sbs_local(i)
        threads.append(threading.Thread(target = sbs_local, args=(i,)))
        threads[-1].start()
    for t in threads: t.join()

    np.save(path+name+'_Coms_tresh.npy', coms_tresh)
    coms_tresh = np.roll(coms_tresh, 2, -1)*np.array([metadata['SizeX'], metadata['SizeY'], metadata['SizeZ']])[np.newaxis, np.newaxis]
    np.save(path+name+'_Coms_tresh_units.npy', coms_tresh)

    np.save(path+name+'_Shapes_Deconv.npy', np.array(deconv_sporos, dtype=object))
    np.save(path+name+'_Shapes_Tresh.npy', np.array(tresh_sporos, dtype=object))
    np.save(path+name+'_Boxpos.npy', box_positions_sporos)


def skin_frame(tsporo, tresh_sporo, metadata):
    """
    Skins the sporo (i.e. reduce thickness to define biocenter) and returns the skinned sporo and the center of mass.
    """
    skinned_x = (np.pad(tsporo[:,:-1], ((0,0),(1,0),(0,0)))<tsporo)*(np.pad(tsporo[:,1:], ((0,0),(0,1),(0,0)))<tsporo)
    skinned_x = np.pad(1*skinned_x[:,:-2], ((0,0),(2,0),(0,0)))+np.pad(2*skinned_x[:,:-1], ((0,0),(1,0),(0,0)))+3*skinned_x+np.pad(2*skinned_x[:,1:], ((0,0),(0,1),(0,0)))+np.pad(1*skinned_x[:,2:], ((0,0),(0,2),(0,0)))
    skinned_y = (np.pad(tsporo[:,:,:-1], ((0,0),(0,0),(1,0)))<tsporo)*(np.pad(tsporo[:,:,1:], ((0,0),(0,0),(0,1)))<tsporo)
    skinned_y = np.pad(1*skinned_y[:,:,:-2], ((0,0),(0,0),(2,0)))+np.pad(2*skinned_y[:,:,:-1], ((0,0),(0,0),(1,0)))+3*skinned_y+np.pad(2*skinned_y[:,:,1:], ((0,0),(0,0),(0,1)))+np.pad(1*skinned_y[:,:,2:], ((0,0),(0,0),(0,2)))
    skinned_z = (np.pad(tsporo[:-1], ((1,0),(0,0),(0,0)))<tsporo)*(np.pad(tsporo[1:], ((0,1),(0,0),(0,0)))<tsporo)
    skinned_z = (np.pad(1*skinned_z[:-2], ((2,0),(0,0),(0,0)))+np.pad(2*skinned_z[:-1], ((1,0),(0,0),(0,0)))
                    +3*skinned_z+np.pad(2*skinned_z[1:], ((0,1),(0,0),(0,0)))+np.pad(1*skinned_z[2:], ((0,2),(0,0),(0,0))))
    skinned = (skinned_x+skinned_y+skinned_z>6)*ndimage.binary_dilation(tresh_sporo, structure=ndimage.generate_binary_structure(3,2))
    skinned = ndimage.binary_closing(np.pad(skinned, ((3,3),(3,3),(3,3)), mode='edge'), structure=ndimage.generate_binary_structure(3,2), iterations=2)[3:-3,3:-3,3:-3]

    com = ndimage.center_of_mass(skinned)
    if skinned.sum()>1:
        ppositions = (np.stack(np.where(skinned), axis=1)-np.array(ndimage.center_of_mass(skinned))[np.newaxis])*np.array([metadata['SizeZ'], metadata['SizeX'], metadata['SizeY']])[np.newaxis]
        eigval, eigvec = np.linalg.eig(np.cov(ppositions.T))
        pdir = eigvec[...,eigval.argmax()]
        biocent = np.mean(ppositions[np.abs(np.dot(pdir, ppositions.T))<1.3], axis=0)/np.array([metadata['SizeZ'], metadata['SizeX'], metadata['SizeY']]) + com
    else:
        biocent = com
    if np.isnan(biocent).any():
        biocent = com
    return skinned, biocent

def skin_sporos(path, name='PC', dname='PC'):
    """
    Skins the sporos (i.e. reduce thickness to define biocenter) in the data in path and saves the results in path.
    """
    boxpos = np.load(path + name + '_Boxpos.npy')
    sporos_tresh = np.load(path + name + '_Shapes_Tresh.npy', allow_pickle=True)
    sporos_deconv = np.load(path + name + '_Shapes_Deconv.npy', allow_pickle=True)
    metadata = np.load(path + dname + '_Metadata.npy', allow_pickle=True).item()
    sporos_skinned = []
    biocents = np.empty_like(boxpos).astype(np.float64)
    biocents.fill(np.nan)
    for s in range(len(sporos_deconv)):
        if sporos_deconv[s] is None:
            sporos_skinned.append(None)
        else:
            skinned_list = []
            for t in range(len(sporos_deconv[s])):
                if sporos_deconv[s][t] is None:
                    skinned_list.append(None)
                else:
                    skinned, biocent = skin_frame(sporos_deconv[s][t], sporos_tresh[s][t], metadata)
                    skinned_list.append(skinned.copy())
                    biocents[s,t] = biocent + boxpos[s,t]
            sporos_skinned.append(skinned_list)
    np.save(path + name + '_Biocents.npy', biocents)
    np.save(path + name + '_Shapes_Skinned.npy', np.array(sporos_skinned, dtype=object))
    biocents = np.roll(biocents, 2, -1)*np.array([metadata['SizeX'], metadata['SizeY'], metadata['SizeZ']])[np.newaxis, np.newaxis]
    np.save(path+name+'_Biocents_units.npy', biocents)

def fit_trajectories(path, epochs=10000, name='PC', dname='PC'):
    """
    Fits the trajectories of the sporos in the data in path with helical segments and saves the results in path.
    """
    biocents = np.load(path+name+'_Biocents_units.npy', allow_pickle=True)
    biocents = biocents * np.array([1,1,0.88]) # Estimated factor RI missmatch z stretch
    metadata = np.load(path + dname + '_Metadata.npy', allow_pickle=True).item()
    framedur = metadata['Delta_T']
    res = fit_all_traj(biocents, epochs, framedur)
    np.save(path+name+'_Helixfits.npy', {'lossval':res[0], 'params':res[1], 'ts_opt':res[2], 'guesshelix':res[3], 'reshelix':res[4], 'reshelix_ts':res[5], 'inverted':res[6]})
