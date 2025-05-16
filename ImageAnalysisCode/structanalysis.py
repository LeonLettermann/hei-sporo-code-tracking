# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tifffile
from ImageAnalysisCode import pvexp
import os

### This is experimental code used for analyzing interplay of sporozoites and structures.

def get_structure_binarized_and_ROI(structure, metadata):
    s_smooth = ndimage.gaussian_filter(structure, sigma=(1,3,3))

    int_measure = np.sort(structure.reshape(structure.shape[0],-1), axis=1)[:,-20000]
    int_top_cutoff = np.argmax(int_measure>0.8*np.nanmax(int_measure))
    #s_smooth[:int_top_cutoff] = 0 Structure are hight -> Disable cutoff??
    #print('First {} frames removed'.format(int_top_cutoff))

    s_smooth = np.pad(s_smooth, ((5,5),(0,0),(0,0)), mode='edge')
    s_background = ndimage.gaussian_filter(s_smooth, sigma=(0,13,13))

    s_binary = (s_smooth>1.3*s_background)*(s_smooth>np.mean(s_smooth))
    s_binary = ndimage.binary_erosion(ndimage.binary_dilation(s_binary, iterations=2), iterations=2)[5:-5]
    s_binary_mapped = ndimage.map_coordinates(s_binary, np.meshgrid(np.linspace(0,s_binary.shape[0]-1,np.round(s_binary.shape[0]*metadata['SizeZ']).astype(int)), 
                                                                    np.linspace(0, s_binary.shape[1]-1, np.round(s_binary.shape[1]*metadata['SizeX']).astype(int)), 
                                                                    np.linspace(0, s_binary.shape[2]-1, np.round(s_binary.shape[2]*metadata['SizeY']).astype(int)), 
                                                                    indexing='ij'), order=0, mode='nearest')

    s_binary_mask = ndimage.gaussian_filter(1.*s_binary_mapped, sigma=(2,20,20))
    s_binary_mask = s_binary_mask>0.12
    s_binary_mask = s_binary_mask*s_binary_mask[-1][np.newaxis]
    #s_binary_mask[:int_top_cutoff] = 0

    return s_binary_mapped, s_binary_mask



def twoend(x):
    return x[:,1:]*x[:,:-1]

def pure(v):
    return np.array([ndimage.label(~np.isnan(v[i]))[1]==1 for i in range(len(v))])

def mask_largest_component(positions):
    mask = np.zeros(positions.shape[:2], dtype=bool)
    for lv in range(positions.shape[0]):
        if np.isnan(positions[lv]).all():
            continue
        label, num = ndimage.label(~np.isnan(positions[lv,:,0]))
        sizes = np.bincount(label.ravel())
        mask[lv][(label==(np.argmax(sizes[1:]))+1)]=True
    return mask

def msd_conditioned(positions, condition):
    positions_h = positions.copy()
    positions_h[~condition] = np.nan
    positions_h[~mask_largest_component(positions_h)]=np.nan
    #positions_h = positions_h[(~np.isnan(positions_h[:,:,0])).sum(axis=1)>9]
    msds = np.empty(positions_h.shape[:2], dtype=float)
    msds.fill(np.nan)
    for lv in range(positions_h.shape[0]):
        argstart = np.argmax(~np.isnan(positions_h[lv,:,2]))
        msds[lv] = np.linalg.norm(np.roll(positions_h[lv], -argstart, axis=0)-positions_h[lv,argstart], axis=-1)**2
    return msds

def analyze_vel_and_msd(path, name='PC', strName='PC'):
    filename = path.split('/')[-2]
    if not os.path.isdir(path + name + '_StrAnalysis'):
        os.mkdir(path + name + '_StrAnalysis')
    outpath = path + name + '_StrAnalysis/'
    biocents = np.load(path + name + '_Biocents_units.npy', allow_pickle=True)
    boxpos = np.load(path + name + '_Boxpos.npy', allow_pickle=True)
    metadata = np.load(path + strName + '_Metadata.npy', allow_pickle=True).item()
    skinned = np.load(path + name + '_Shapes_Skinned.npy', allow_pickle=True)
    structure = tifffile.imread(path + strName + '_Background_corr.tif')

    steps = np.diff(biocents, axis=1)
    normsteps = np.linalg.norm(steps, axis=2)
    cumlength = np.nansum(normsteps, axis=1)
    totdist = np.linalg.norm(np.nanmax(biocents, axis=1)-np.nanmin(biocents, axis=1), axis=1)
    valid = (cumlength>3)
    biocents = biocents[valid]
    boxpos = boxpos[valid]
    skinned = skinned[valid]

    framedur = metadata['Delta_T']

    s_binary_mapped, s_binary_mask = get_structure_binarized_and_ROI(structure, metadata)
    plt.imshow(np.sum(s_binary_mask, axis=0)+np.sum(s_binary_mapped, axis=0))
    plt.savefig(outpath + 'PV_StructureMask.png', bbox_inches='tight')
    plt.close()

    s_binary_mask_F = np.moveaxis(s_binary_mask*1,0,2)
    substclose = (biocents[:,:,2]>(np.nanmax(biocents[:,:,2])-3))
    in_roi = np.array([[s_binary_mask_F[tuple(np.clip(np.round(biocents[i,j]),0,np.array(s_binary_mask_F.shape)-1).astype(int))] if not np.isnan(biocents[i,j,2]) else False for j in range(biocents.shape[1])] for i in range(len(biocents))]).astype(bool)

    pvexp.export_tracks(biocents, outpath + 'PV_Tracks', extradata = {'in_roi':1*in_roi, 'substclose':1*substclose})
    pvexp.export_stillframe(1*s_binary_mapped, outpath + 'PV_Structure')
    pvexp.export_stillframe(s_binary_mask_F, outpath + 'PV_StructMask')
    if not os.path.exists(outpath + 'PV_Skinned'):
        os.mkdir(outpath + 'PV_Skinned')
    pvexp.export_pointsets(biocents, boxpos, skinned, metadata, outpath + 'PV_Skinned/')

    velocities = np.linalg.norm(np.diff(biocents, axis=1),axis=2)/framedur
    velocities_2D = np.where(twoend(substclose), velocities, np.nan)
    velocities_3D = np.where(twoend(~substclose), velocities, np.nan)
    velocities_roi = np.where(twoend(in_roi), velocities, np.nan)
    velocities_no_roi = np.where(twoend(~in_roi), velocities, np.nan)
    velocities_roi_2D = np.where(twoend(np.logical_and(in_roi, substclose)), velocities, np.nan)
    velocities_roi_3D = np.where(twoend(np.logical_and(in_roi, ~substclose)), velocities, np.nan)
    velocities_no_roi_2D = np.where(twoend(np.logical_and(~in_roi, substclose)), velocities, np.nan)
    velocities_no_roi_3D = np.where(twoend(np.logical_and(~in_roi, ~substclose)), velocities, np.nan)
    datasets = [velocities, velocities_2D, velocities_3D, velocities_roi, velocities_no_roi, velocities_roi_2D, velocities_roi_3D, velocities_no_roi_2D, velocities_no_roi_3D]
    names = ['all', '2D', '3D', 'in ROI', 'no ROI', '2D in ROI', '3D in ROI', '2D no ROI', '3D no ROI']

    fig, ax = plt.subplots(1,1)
    datah = [v[~np.isnan(v)] if (~np.isnan(v)).any() else np.ones(2) for v in datasets]
    ax.violinplot(datah, showmedians=True, 
                    positions = np.arange(9), widths = 0.8);
    ax.set_xticks(np.arange(9), names, rotation=45);
    for i in range(9):
        ax.text(i, 1.1*np.nanmax(np.concatenate(datah)), str(len(datah[i])), ha='center')
    plt.savefig(outpath + 'Velocities.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,1)
    datah = [np.nanmean(v,axis=1)[pure(v)] if (pure(v)).any() else np.ones(2) for v in datasets]
    ax.violinplot(datah, showmedians=True, 
                    positions = np.arange(9), widths = 0.8);
    ax.set_xticks(np.arange(9), names, rotation=45);
    for i in range(9):
        ax.text(i, 1.1*np.nanmax(np.concatenate(datah)), str(len(datah[i])), ha='center')
    plt.savefig(outpath + 'VelocitiesPerSporo.png', bbox_inches='tight')
    plt.close()

    msds = msd_conditioned(biocents, np.ones(biocents.shape[:2], dtype=bool))
    msds_2D = msd_conditioned(biocents, substclose)
    msds_3D = msd_conditioned(biocents, ~substclose)
    msds_roi = msd_conditioned(biocents, in_roi)
    msds_no_roi = msd_conditioned(biocents, ~in_roi)
    msds_roi_2D = msd_conditioned(biocents, np.logical_and(in_roi, substclose))
    msds_roi_3D = msd_conditioned(biocents, np.logical_and(in_roi, ~substclose))
    msds_no_roi_2D = msd_conditioned(biocents, np.logical_and(~in_roi, substclose))
    msds_no_roi_3D = msd_conditioned(biocents, np.logical_and(~in_roi, ~substclose))
    msd_coll = [msds, msds_2D, msds_3D, msds_roi, msds_no_roi, msds_roi_2D, msds_roi_3D, msds_no_roi_2D, msds_no_roi_3D]

    for i in range(9):
        plt.plot(np.nanmean(msd_coll[i],axis=0), label=names[i])
    plt.legend()
    plt.savefig(outpath + 'MSDs.png', bbox_inches='tight')
    plt.close()

    resdict = {'names':names, 'biocents':biocents, 'substclose':substclose, 'in_roi':in_roi, 'msds':msd_coll, 'velocities':datasets, 'stucture':s_binary_mapped, 'structure_mask':s_binary_mask_F, 'metadata':metadata}
    np.save(outpath + 'TrackResults.npy', resdict)

