import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.example_libraries.optimizers as jaxopt
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
import ImageAnalysisCode as iac
from scipy import ndimage, stats, optimize, signal
import os
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (0, 1, 0)]  # Black to green
cmap1 = LinearSegmentedColormap.from_list('black_to_green', colors)
colors = [(0, 0, 0), (1, 0, 0)]  # Black to green
cmap2 = LinearSegmentedColormap.from_list('black_to_red', colors)
import matplotlib.animation as animation
from matplotlib.patches import Arc
from matplotlib.cm import viridis
import trackpy as tp
import pandas as pd

import threading

def nanfree(x):
    return x[~np.isnan(x)]

### First analysis part (Sporozoite tracking and bead tracking)

def tresh_and_track_here(data_np, metadata, base_treshold=15, dilate = 0, max_reconn_dist = 10, min_size=20):
    """Threshold and track the sporozoites."""
    original = data_np.copy()
    if len(data_np.shape)==3:
        data_np = data_np[:,np.newaxis]
    data_np = ((data_np.astype(np.float32)*255)/data_np.max()).astype(np.uint8)
    data_np = iac.track.thresh_n_label(data_np, ada_thresh_width = 11, ada_thresh_strength=0, abs_thresh=base_treshold, min_size=min_size, max_size=5000, min_size_per_z=0, max_size_per_z=5000, co_mask=None, dilate=dilate)
    data_np = iac.track.track(data_np, metadata, max_reconn_dist, min_size=min_size)
    if len(original.shape)==3:
        original = original[:,np.newaxis]
    maxlabel = data_np.max()
    tresh_sporos = [None]*(maxlabel+1)
    deconv_sporos = [None]*(maxlabel+1)
    box_positions_sporos = -np.ones(((maxlabel+1), data_np.shape[0],3), dtype=np.int16)
    coms_tresh = np.zeros(((maxlabel+1), data_np.shape[0],3), dtype=np.float16)*np.nan

    threads = []
    sbs_local = lambda i: iac.pipelines.save_boxed_sporo(i, data_np, original, tresh_sporos, deconv_sporos, box_positions_sporos, coms_tresh)
    for i in np.arange(maxlabel)+1:
        # sbs_local(i)
        threads.append(threading.Thread(target = sbs_local, args=(i,)))
        threads[-1].start()
    for t in threads: t.join()
    coms_tresh = np.roll(coms_tresh, 2, -1)*np.array([metadata['SizeX'], metadata['SizeY'], metadata['SizeZ']])[np.newaxis, np.newaxis]
    return data_np, coms_tresh, deconv_sporos, tresh_sporos, box_positions_sporos


def identify_paired_trajectories(track_intl, track1, track2, loc1_intl):
    """Between the two different planes in which beads are tracked, pair up trajectories showing the same bead."""
    identifies_pairs = []
    unidentified = []
    for i in np.unique(track_intl['particle']):
        part_in_intl = track_intl[track_intl['particle'] == i].index
        part_1_in_intl = part_in_intl[part_in_intl < len(loc1_intl)]
        part_2_in_intl = part_in_intl[part_in_intl >= len(loc1_intl)]
        particles_in_1, counts_1 = np.unique(track1.loc[part_1_in_intl, 'particle'], return_counts=True)
        particles_in_2, counts_2 = np.unique(track2.loc[part_2_in_intl, 'particle'], return_counts=True)
        if counts_1.sum()<20 or counts_2.sum()<20:
            continue
        if len(counts_1) == 1 and len(counts_2) == 1:
            identifies_pairs.append([particles_in_1[0], particles_in_2[0]])
        elif ((len(counts_1) == 1 or counts_1.max()>5*np.sort(counts_1)[-2])
            and (len(counts_2) == 1 or counts_2.max()>5*np.sort(counts_2)[-2])):
            identifies_pairs.append([particles_in_1[np.argmax(counts_1)], particles_in_2[np.argmax(counts_2)]])
        else:
            unidentified.append(i)
    identifies_pairs = np.array(identifies_pairs)
    _, inv1, c1 = np.unique(identifies_pairs[:,0], return_counts=True, return_inverse=True)
    _, inv2, c2 = np.unique(identifies_pairs[:,1], return_counts=True, return_inverse=True)
    identifies_pairs = identifies_pairs[np.logical_and((c1==1)[inv1], (c2==1)[inv2])]
    return identifies_pairs, unidentified

def compute_massmeans(track1, track2):
    """Compute the mean mass of the beads in each track."""
    massmeans = np.zeros((track1['particle'].max()+1,2), dtype=float)
    massmeans.fill(np.nan)
    for i in np.unique(track1['particle']):
        massmeans[i]=((np.nanmean(track1['mass'][track1['particle'] == i]),np.nanmean(track2['mass'][track2['particle'] == i])))
    return massmeans

def fill_traj(traj, maxlen):
    trajN = np.zeros((maxlen, 2))
    trajN.fill(np.nan)
    trajN[traj[:,2].astype(int)] = traj[:,:2]
    return trajN

def track_beads(frames_1, frames_2):
    """Track the beads in the two different planes."""
    loc1 = tp.batch(frames_1, 5, percentile=20, separation=2, maxsize=2)
    loc2 = tp.batch(frames_2, 5, percentile=20, separation=2, maxsize=2)
    loc2.index = loc2.index + len(loc1)
    track1 = tp.link(loc1, 1.5, memory=3)
    track2 = tp.link(loc2, 1.5, memory=3)
    loc1_intl = loc1.copy()
    loc2_intl = loc2.copy()
    loc1_intl['frame'] = loc1_intl['frame']*2
    loc2_intl['frame'] = loc2_intl['frame']*2+1
    track_intl = tp.link(pd.concat([loc1_intl,loc2_intl]), 1.5, memory=3)
    identifies_pairs, unidentified = identify_paired_trajectories(track_intl, track1, track2, loc1_intl)

    track1id = track1[np.in1d(track1['particle'].values, identifies_pairs[:,0])]
    track2id = track2[np.in1d(track2['particle'].values, identifies_pairs[:,1])]
    pairmap = np.zeros(identifies_pairs[:,1].max()+1, dtype=int)
    pairmap[identifies_pairs[:,1]] = identifies_pairs[:,0]
    track2id['particle'] = pairmap[track2id['particle'].values]
    massmeans = compute_massmeans(track1id, track2id)
    in_zlayer_1 = (massmeans[:,0]>massmeans[:,1])
    traj1 = [np.array(track1id[['x','y', 'frame']][track1id['particle']==i]) for i in track1id['particle'].unique() if in_zlayer_1[i]]
    traj1 = np.array([fill_traj(t,len(frames_1)) for t in traj1])
    traj2 = [np.array(track2id[['x','y', 'frame']][track2id['particle']==i]) for i in track2id['particle'].unique() if not in_zlayer_1[i]]
    traj2 = np.array([fill_traj(t,len(frames_2)) for t in traj2])
    return traj1, traj2, massmeans, identifies_pairs, track1id, track2id


def comp_displacements(traj1, traj2):
    """Compute the displacements of the beads in the two different planes."""
    beadmean1, beadmean2 = np.nanmean(traj1, axis=1), np.nanmean(traj2, axis=1)
    disp1, disp2 = traj1-beadmean1[:,None], traj2-beadmean2[:,None]
    disp1_sm, disp2_sm = ndimage.gaussian_filter(disp1, (0,2,0)), ndimage.gaussian_filter(disp2, (0,2,0))
    shift_corr = ndimage.gaussian_filter(np.nanmean(np.concatenate([disp1_sm, disp2_sm],axis=0), axis=0), (5,0), mode='nearest')
    beadmean1, beadmean2 = np.nanmean(traj1-shift_corr[None,:], axis=1), np.nanmean(traj2-shift_corr[None,:], axis=1)
    disp1, disp2 = traj1 - shift_corr[None,:]-beadmean1[:,None], traj2 - shift_corr[None,:]-beadmean2[:,None]
    disp1_sm, disp2_sm = ndimage.gaussian_filter(disp1, (0,2,0)), ndimage.gaussian_filter(disp2, (0,2,0))
    return disp1, disp2, disp1_sm, disp2_sm, beadmean1, beadmean2, shift_corr

def full_analysis(in_file, out_file=None, series=0, returning=False):
    """Full analysis of the sporozoites and beads."""
    data, metadata = iac.general.load_bioformat(in_file, series)
    data_np = data.to_numpy()[:,:,:]
    sporo = data_np[:,0]
    beads = data_np[:,1]

    # Bead Analysis
    frames_1 = ndimage.median_filter(beads[:,0], (3,1,1))
    frames_2 = ndimage.median_filter(beads[:,1], (3,1,1))
    traj1, traj2, massmeans, identifies_pairs, track1id, track2id = track_beads(frames_1, frames_2)
    disp1, disp2, disp1_sm, disp2_sm, beadmean1, beadmean2, shift_corr = comp_displacements(traj1, traj2)

    # Sporo Analysis
    labeled1, coms_tresh1, deconv_sporos1, tresh_sporos1, box_positions_sporos1 = tresh_and_track_here(sporo[:,:1]-sporo[:,:1].min(), metadata)
    labeled2, coms_tresh2, deconv_sporos2, tresh_sporos2, box_positions_sporos2 = tresh_and_track_here(sporo[:,1:2]-sporo[:,1:2].min(), metadata)

    resdict = {'beads':beads, 'sporo':sporo, 'frames_1':frames_1, 'frames_2':frames_2, 'metadata':metadata,
                'traj1':traj1, 'traj2':traj2, 'massmeans':massmeans, 'identifies_pairs':identifies_pairs, 'track1id':track1id, 'track2id':track2id, 
               'disp1':disp1, 'disp2':disp2, 'disp1_sm':disp1_sm, 'disp2_sm':disp2_sm, 'beadmean1':beadmean1, 'beadmean2':beadmean2, 'shift_corr':shift_corr, 
               'labeled1':labeled1, 'coms_tresh1':coms_tresh1, 'deconv_sporos1':deconv_sporos1, 'tresh_sporos1':tresh_sporos1, 'box_positions_sporos1':box_positions_sporos1, 
               'labeled2':labeled2, 'coms_tresh2':coms_tresh2, 'deconv_sporos2':deconv_sporos2, 'tresh_sporos2':tresh_sporos2, 'box_positions_sporos2':box_positions_sporos2}
    
    if not out_file is None:
        np.save(out_file, resdict)

    if returning:
        return resdict
    
def animate_results(frames_1, frames_2, sporo, beadmean1, beadmean2, disp1_sm, disp2_sm, savefile, xlim=None, ylim=None):
    """Animate the results of the analysis."""
    f = 0
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    im = ax.imshow(np.maximum(frames_1[f],frames_2[f]), cmap='gray', vmax = 15000)
    sporohere = np.max(sporo[f], axis=0)
    sporohere = np.clip(sporohere, np.quantile(sporohere, 0.5), np.inf)
    sporohere = sporohere - sporohere.min()
    alpha_co = np.quantile(sporo, 0.98)
    im2 = ax.imshow(sporohere, cmap='viridis', alpha = np.clip(sporohere, 0, alpha_co)/alpha_co)
    ax.scatter(beadmean1[:,0], beadmean1[:,1], edgecolor='r', marker='o', s=10, facecolor='none', lw=0.4)
    ax.scatter(beadmean2[:,0], beadmean2[:,1], edgecolor='g', marker='o', s=10, facecolor='none', lw=0.4)
    qv1 = ax.quiver(beadmean1[:,0], beadmean1[:,1], disp1_sm[:,f,0], disp1_sm[:,f,1], color='r', scale=1e-1, scale_units='xy')
    qv2 = ax.quiver(beadmean2[:,0], beadmean2[:,1], disp2_sm[:,f,0], disp2_sm[:,f,1], color='g', scale=1e-1, scale_units='xy')

    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim[::-1])
    def update(f):
        sporohere = np.max(sporo[f], axis=0)
        sporohere = np.clip(sporohere, np.quantile(sporohere, 0.5), np.inf)
        sporohere = sporohere - sporohere.min()
        alpha_co = np.quantile(sporo, 0.98)
        im.set_data(np.maximum(frames_1[f],frames_2[f]))
        im2.set_data(sporohere)
        im2.set_alpha(np.clip(sporohere, 0, alpha_co)/alpha_co)
        qv1.set_UVC(disp1_sm[:,f,0], disp1_sm[:,f,1])
        qv2.set_UVC(disp2_sm[:,f,0], disp2_sm[:,f,1])
        return im, qv1, qv2
    ani = animation.FuncAnimation(fig, update, frames=len(frames_1), interval=100, blit=True)
    ani.save(savefile, writer='ffmpeg', fps=10)


## Second analysis Part (Traction force inference based on displacements)

def stat_test(data1, data2):
    d1flat = data1.flatten()
    d1flat = d1flat[~np.isnan(d1flat)]
    d2flat = data2.flatten()
    d2flat = d2flat[~np.isnan(d2flat)]
    return stats.mannwhitneyu(d1flat, d2flat)

def pc_update_bm_and_disp(beadmean_in, traj_in, labeled, median_size = 31):
    """Update the beadmean and displacements based on average bead positions weighted by sporozoite positions."""
    bead_sporo_dist1 = np.nanmin(np.linalg.norm(beadmean_in-np.stack(np.where((labeled[:,0]>0).any(axis=0)), axis=-1)[:,None],axis=-1), axis=0)
    traj = traj_in - np.nanmean((traj_in-np.nanmean(traj_in, axis=1)[:,None])[bead_sporo_dist1>50],axis=0)[None]

    bead_sporo_dist1 = np.array([np.nanmin(np.linalg.norm(traj[:,f][None]-np.stack(np.where(labeled[f,0]>0), axis=-1)[:,None],axis=-1), axis=0) for f in range(len(labeled))]).T
    weights_bs_dist1 = bead_sporo_dist1**2/np.nansum(bead_sporo_dist1**2, axis=1)[:,None]
    beadmean = np.nansum(traj*weights_bs_dist1[:,:,None], axis=1)
    disp_sm = traj-beadmean[:,None]
    disp_sm = ndimage.gaussian_filter1d(disp_sm, 1, axis=1)
    disp_sm = disp_sm - ndimage.median_filter(disp_sm, size=(1,median_size,1), mode='reflect')
    return beadmean, disp_sm

def comp_biocent(tresh, metadata):
    """Compute the biocenter of the sporozoite, as well as the apical and basal tips."""
    com = ndimage.center_of_mass(tresh)
    if tresh.sum()>1:
        ppositions = (np.stack(np.where(tresh), axis=1)-np.array(com)[np.newaxis])*np.array([metadata['SizeZ'], metadata['SizeX'], metadata['SizeY']])[np.newaxis]
        eigval, eigvec = np.linalg.eig(np.cov(ppositions.T))
        pdir = eigvec[...,eigval.argmax()]
        biocent = np.mean(ppositions[np.abs(np.dot(pdir, ppositions.T))<1.3], axis=0)/np.array([metadata['SizeZ'], metadata['SizeX'], metadata['SizeY']]) + com
        
        coords = np.stack(np.where(tresh), axis=1)
        bc_dists = np.linalg.norm(coords-np.array(biocent)[np.newaxis], axis=1)
        tip = coords[bc_dists.argmax()]
        tip_dists = np.linalg.norm(coords-tip, axis=1)
        tip2 = coords[np.argmax(np.minimum(tip_dists, bc_dists))]
    else:
        biocent = com
        tip, tip2 = np.nan*np.array(com), np.nan*np.array(com)
    if np.isnan(biocent).any():
        biocent = com
    return biocent, tip, tip2

def comp_biocent_vel(biocents):
    """Compute the velocity of the biocenter."""
    biocent_vel1 = np.pad(biocents[:,:,0], ((0,0),(1,1),(0,0)), mode='edge')
    biocent_vel1 = (biocent_vel1[:,2:]-biocent_vel1[:,:-2])/2
    biocent_vel1_sm = ndimage.gaussian_filter1d(biocent_vel1, 1, axis=1, mode='nearest')
    return biocent_vel1_sm

def comp_ordered_biocents(biocents, biocent_vel):
    """Order the biocenters based on the velocity of the biocenter (i.e. assign front and back)."""
    tip1align = np.einsum('ijk,ijk->ij', biocent_vel/np.linalg.norm(biocent_vel, axis=-1)[:,:,None], (biocents[:,:,1]-biocents[:,:,0])/np.linalg.norm(biocents[:,:,1]-biocents[:,:,0], axis=-1)[:,:,None])
    tip2align = np.einsum('ijk,ijk->ij', biocent_vel/np.linalg.norm(biocent_vel, axis=-1)[:,:,None], (biocents[:,:,2]-biocents[:,:,0])/np.linalg.norm(biocents[:,:,2]-biocents[:,:,0], axis=-1)[:,:,None])
       
    biocents_ordered = np.stack([biocents[:,:,0],
                                np.where((tip1align>tip2align)[:,:,None], biocents[:,:,1], biocents[:,:,2]),
                                np.where((tip1align>tip2align)[:,:,None], biocents[:,:,2], biocents[:,:,1])], axis=-2)
    return biocents_ordered

def pc_get_biocents_and_vels(tresh_sporos, box_positions_sporos, metadata):
    """Compute the biocenters and velocities of the sporozoites."""
    biocents = np.array([np.array([np.array(comp_biocent(t,metadata))[:,2:0:-1] if not t is None else np.nan*np.ones((3,2)) for t in ts]) if not ts is None else np.nan*np.ones((metadata['n_T'],3,2)) for ts in tresh_sporos])
    biocents = biocents + box_positions_sporos[:,:,np.newaxis,2:0:-1]
    biocent_vel_sm = comp_biocent_vel(biocents)
    biocents_ordered = comp_ordered_biocents(biocents, biocent_vel_sm)
    return biocents_ordered, biocent_vel_sm

def clockwise_scors(biocent_ordered_in):
    """Compute the clockwise score of the biocenters."""
    tb_t= biocent_ordered_in - np.nanmean(biocent_ordered_in, axis=1)[:,None,:]
    tb_step = np.diff(tb_t, axis=1)
    tb_step = tb_step/np.linalg.norm(tb_step, axis=-1)[:,:,None]
    tb_posnorm = (tb_t[:,:-1]+tb_t[:,1:])/2
    tb_posnorm = tb_posnorm/np.linalg.norm(tb_posnorm, axis=-1)[:,:,None]
    crossvecs = np.cross(tb_posnorm, tb_step, axisa=-1, axisb=-1, axisc=-1)
    return np.nanmean(crossvecs, axis=1)

def pc_average_disp_over_beads(beadmean, disp_sm, sigma=5., maxdisp=0.8):
    """Average the displacements over the beads based on their distances."""
    beaddists = np.linalg.norm(beadmean[:,None]-beadmean[None], axis=-1)
    weights = np.nan_to_num(np.exp(-beaddists**2/(2*sigma**2)))
    weights = weights/weights.sum(axis=1)[:,None]
    disp_averaged = np.stack([np.nansum((disp_sm[:,t]/np.clip(np.linalg.norm(disp_sm[:,t], axis=-1)/maxdisp,1.,np.inf)[:,None])[None]*weights[:,:,None], axis=1) for t in range(disp_sm.shape[1])], axis=1)
    return disp_averaged, beaddists

def identify_paired_sporos(labeled1,labeled2):
    """Identify which sporozoites labeled and tracked independently in the two planes are the same."""
    identified_pairs = []
    for l1 in np.unique(labeled1):
        partners, counts = np.unique(labeled2[labeled1 == l1], return_counts=True)
        if 0 in partners:
            counts = counts[~(partners == 0)]
            partners = partners[~(partners == 0)]
        if len(partners) == 1:
            identified_pairs.append([l1, partners[0]])
        elif len(partners) > 1:
            if counts.max()>counts.sum()*0.5:
                identified_pairs.append([l1, partners[counts.argmax()]])
        else:
            print('Unidentified pair {}, partners {}, counts {}'.format(l1, partners, counts))
    identified_pairs = np.array(identified_pairs)
    return identified_pairs

def comp_spine_circle(points):
    """Compute the circle defined by the three sporozoite points."""
    matrix = np.concatenate([2*points, np.ones((len(points),3,1))], axis=-1)
    sol_abc = np.linalg.solve(matrix, -(points**2).sum(axis=-1))
    circ_cent = -sol_abc[:,:2]
    circ_rads = np.sqrt(sol_abc[:,0]**2+sol_abc[:,1]**2-sol_abc[:,2])
    return circ_cent, circ_rads

def build_radial_rep(biocents_i, disp_sm, beadmean, dist_to_take=30.):
    """Build the radial representation of displacements in a polar coordinate system based on the sporozoite biocenter."""
    tcdists = np.linalg.norm(biocents_i[None]-beadmean[:,None,None], axis=-1)
    valid = tcdists.min(axis=-1)<dist_to_take
    d_here, b_here = disp_sm[valid.any(axis=1)], beadmean[valid.any(axis=1)]
    circ_cent, circ_rad = comp_spine_circle(biocents_i)
    front_vecs = (biocents_i[:,1]-circ_cent)/np.linalg.norm(biocents_i[:,1]-circ_cent, axis=-1)[:,None]
    cent_vecs = (biocents_i[:,0]-circ_cent)/np.linalg.norm(biocents_i[:,0]-circ_cent, axis=-1)[:,None]
    back_vecs = (biocents_i[:,2]-circ_cent)/np.linalg.norm(biocents_i[:,2]-circ_cent, axis=-1)[:,None]
    opening_angle = np.arccos(np.sum(front_vecs*back_vecs, axis=-1))
    front_cent_angle = -np.arccos(np.sum(front_vecs*cent_vecs, axis=-1))*np.sign(np.cross(cent_vecs,front_vecs))
    bead_vecs = ((b_here[:,None]-circ_cent[None])/np.linalg.norm(b_here[:,None]-circ_cent[None], axis=-1)[:,:,None])
    disp_vecs = d_here/np.linalg.norm(d_here, axis=-1)[:,:,None]
    rads_here = np.linalg.norm(b_here[:,None]-circ_cent[None], axis=-1)/circ_rad[None]
    ang_here = -np.arccos(np.sum(bead_vecs*cent_vecs[None], axis=-1))*np.sign(np.cross(cent_vecs[None],bead_vecs))/opening_angle[None] # Ahhh
    disp_ang = -np.arccos(np.sum(disp_vecs*cent_vecs[None], axis=-1))*np.sign(np.cross(cent_vecs[None],disp_vecs)) # Ahhh
    disp_mag = np.linalg.norm(d_here, axis=-1)
    return circ_rad, opening_angle, front_cent_angle, rads_here, ang_here, disp_ang, disp_mag, valid

def single_sporo_pc_analysis(isporo, identified_pairs, biocents_ordered, disp_sm, beadmean, dist_to_take=30., sigma_map=2., maxd_map=0.5, frames_to_take=None):
    """Perform the analysis of a single sporozoite."""
    biocents_i = biocents_ordered[identified_pairs[isporo-1]]
    if not frames_to_take is None:
        disp_sm = disp_sm[:,frames_to_take]
        biocents_i = biocents_i[frames_to_take]
    circ_rad, opening_angle, front_cent_angle, rads_here, ang_here, disp_ang, disp_mag, valid = build_radial_rep(biocents_i, disp_sm, beadmean, dist_to_take=dist_to_take)

    mean_rad, mean_opening_angle, mean_f_c_angle = np.nanmedian(circ_rad, axis=0), np.nanmedian(opening_angle, axis=0), np.nanmedian(front_cent_angle, axis=0)
    valid_here = valid[valid.any(axis=1)]
    coords_here = mean_rad*rads_here[valid_here][:,None]*np.stack([np.cos(ang_here[valid_here]*mean_opening_angle), np.sin(ang_here[valid_here]*mean_opening_angle)], axis=-1)
    disps_here = disp_mag[valid_here][:,None]*np.stack([np.cos(disp_ang[valid_here]), np.sin(disp_ang[valid_here])], axis=-1)

    beaddists = np.linalg.norm(coords_here[:,None]-coords_here[None], axis=-1)
    weights = np.nan_to_num(np.exp(-beaddists**2/(2*sigma_map**2)))
    weights = weights/weights.sum(axis=1)[:,None]
    disps_here_av = np.nansum((disps_here/np.clip(np.linalg.norm(disps_here, axis=-1)/maxd_map,1.,np.inf)[:,None])[None]*weights[:,:,None], axis=1)

    return coords_here, disps_here, disps_here_av, valid, mean_f_c_angle, mean_opening_angle, mean_rad, rads_here, ang_here, disp_ang, disp_mag

def plot_map(coords_map, disps_map_av, mean_f_c_angle, mean_opening_angle, mean_rad, savename=None, ax=None, arr_col='g', scale=3.):
    """Plot the map of the displacements."""
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,5))
    angles = (mean_f_c_angle*180/np.pi, -(np.sign(mean_f_c_angle)*mean_opening_angle-mean_f_c_angle)*180/np.pi)
    if mean_f_c_angle<0:
        angles = angles[::-1]
    sporoarc = Arc((0,0),2*mean_rad, 2*mean_rad, theta2=angles[0], theta1=angles[1], 
                    color=viridis(1.), zorder=2, lw=10, alpha=0.7, capstyle='round')
    front = np.array([mean_rad*np.cos(mean_f_c_angle), mean_rad*np.sin(mean_f_c_angle)])
    ax.add_patch(sporoarc)
    ax.scatter(coords_map[:,0], coords_map[:,1], facecolors='none', edgecolors='k', s=5, alpha=0.3)
    ax.scatter(front[0], front[1], c='b', s=30)
    ax.quiver(coords_map[:,0], coords_map[:,1], disps_map_av[:,0], disps_map_av[:,1], scale=scale, color=arr_col, angles='xy')
    ax.autoscale()
    ax.set_aspect('equal')
    if savename is not None:
        plt.savefig(savename, dpi=400)


### Force inference

def g_kernel_herz(x,y,nu,E, plat_scale):
    """Green's function for the Herzian contact model."""
    a = plat_scale
    r = jax.numpy.sqrt(x**2+y**2)
    prefact = 2*(1+nu)/(8*E*a**3)
    w1 = jax.numpy.where(r<=a, 1/4*4*(2-nu)*a**2, 
                         (2-nu)/np.pi*((2*a**2-r**2)*jax.numpy.arcsin(a/r)+a*r*jax.numpy.sqrt(1-a**2/r**2)))
    w2 = jax.numpy.where(r<=a, -1/4*((4-3*nu)*x**2 + (4-nu)*y**2), 
                         nu/2/np.pi*(r**2*jax.numpy.arcsin(a/r)+(2*a**2-r**2)*a/r*jax.numpy.sqrt(1-a**2/r**2))*(x**2-y**2)/r**2)
    w3 = jax.numpy.where(r<=a, 2/4*nu*x*y, 
                         1/np.pi*(r**2*jax.numpy.arcsin(a/r)+(2*a**2-r**2)*a/r*jax.numpy.sqrt(1-a**2/r**2))*(x*y)/r**2)
    w4 = jax.numpy.where(r<=a, -1/4*((4-3*nu)*y**2 + (4-nu)*x**2), 
                         nu/2/np.pi*(r**2*jax.numpy.arcsin(a/r)+(2*a**2-r**2)*a/r*jax.numpy.sqrt(1-a**2/r**2))*(y**2-x**2)/r**2)
    matrix = jax.numpy.array([[w1+w2, w3], [w3, w1+w4]])
    return prefact*matrix

def g_kernel_deriv(x,y,nu,E, plat_scale):
    """Derivative of Green's function for the Herzian contact model used for dipoles."""
    r = jax.numpy.sqrt(x**2+y**2)
    prefact = (1+nu)/(np.pi * E * jax.numpy.where(r>plat_scale, r**5, (plat_scale**3/2+r**4/plat_scale/2)*r**2))

    matrixdx = jax.numpy.array([([-(x * (x ** 2 + (1 -3 * nu) * (y ** 2))), nu * y * (-2 * (x ** 2) + y ** 2)]), ([nu * y * (-2 * (x ** 2) + y ** 2), (-1 + nu) * (x ** 3) -((1 + 2 * nu) * x * (y ** 2))])])
    matrixdy = jax.numpy.array([([(-1 + nu) * (y ** 3) -((x ** 2) * (y + 2 * nu * y)), nu * x * (x ** 2 -2 * (y ** 2))]), ([nu * x * (x ** 2 -2 * (y ** 2)), -(y * ((1 -3 * nu) * (x ** 2) + y ** 2))])])
    return prefact * jax.numpy.stack((matrixdx, matrixdy), axis=1)

def build_f_dip_matrix(f_dips, f_coords):
    """Build a dipole matrix only alowing for dipoles orthogonal to the sporozoite arc."""
    dirs = f_coords/jax.numpy.linalg.norm(f_coords, axis=-1)[:,None]
    return dirs[:,None]*dirs[:,:,None]*f_dips[:,None,None]

def u_single(u_coord, f_coords, f_vecs, f_dips, nu, E, plat_scale):
    """Compute the displacement field predicted by the forces at a given point."""
    deltas = u_coord[None] - f_coords
    dip_matrix = build_f_dip_matrix(f_dips, f_coords)
    us_f = jax.numpy.einsum('fij,fj->fi', jax.vmap(g_kernel_herz, in_axes=(0,0,None,None,None))(deltas[:,0], deltas[:,1], nu, E, plat_scale[0]), f_vecs)
    us_d = jax.numpy.einsum('fikj,fkj->fi', jax.vmap(g_kernel_deriv, in_axes=(0,0,None,None,None))(deltas[:,0], deltas[:,1], nu, E, plat_scale[1]), dip_matrix)
    us = us_f + us_d
    return jax.numpy.sum(us, axis=0)

u_total = jax.vmap(u_single, in_axes=(0,None,None,None,None,None,None))

@jax.jit
def minimizee_sing(f_vecs, u_coords, u_measured, f_coords, nu, E):
    """Loss function for a single plane."""
    u_calc = u_total(u_coords, f_coords, f_vecs.reshape(-1,2), nu, E)
    return jax.numpy.mean((jax.numpy.linalg.norm(u_calc-u_measured, axis=-1)**2))+ 10*jax.numpy.linalg.norm(f_vecs.reshape(-1,2).sum(axis=0))**2


@partial(jax.jit, static_argnums=(7,8,9,10))
def minimizee_sw(params, u_coords1, u_measured1, u_coords2, u_measured2, f_coords1, f_coords2, nu, E, reg_lamb, plat_scale):
    """Loss function for simultaneous minimization of both planes."""
    nuh = nu#f_vecs[-1]
    u_calc1 = u_total(u_coords1, f_coords1, params['f1'], params['d1'], nuh, E, (plat_scale[0], params['plat_scale_d1']))
    u_calc2 = u_total(u_coords2, f_coords2, params['f2'], params['d2'], nuh, E, (plat_scale[0], params['plat_scale_d2']))
    return (jax.numpy.mean((u_calc1-u_measured1)**2)
            + jax.numpy.mean((u_calc2-u_measured2)**2)
            + reg_lamb[0]*jax.numpy.mean(jax.numpy.array([f*(jax.numpy.linalg.norm(params[k], axis=-1) if k[0]=='f' else params[k])**2 for k,f in zip(['f1','f2', 'd1', 'd2'],[1,1,1,1])]))
            + reg_lamb[1]*jax.numpy.mean(jax.numpy.array([f*jax.numpy.sum(jax.numpy.diff(params[k],axis=0)**2) for k,f in zip(['f1','f2', 'd1', 'd2'],[1,1,1,1])]))
            + reg_lamb[2]*(jax.numpy.sum((params['f1'].sum(axis=0)+params['f2'].sum(axis=0))**2) 
                           + jax.numpy.sum((jax.numpy.cross(f_coords1/jax.numpy.linalg.norm(f_coords1, axis=-1)[:,None], params['f1'])
                                            +jax.numpy.cross(f_coords2/jax.numpy.linalg.norm(f_coords2, axis=-1)[:,None], params['f2'])))**2)
            )

def tf_on_testsporo(testsporo, identified_pairs, biocents1_ordered, disp1_sm, beadmean1,biocent_vel1_sm, biocents2_ordered, disp2_sm, beadmean2, biocent_vel2_sm, 
                    frames_1, frames_2, relabeled_both, cwscore, plotname=None, savename=None, sporo_far_dists = 2, max_far_dist = 15,
                    gridres=2., sigmagrid =2., maxdispgrid=1., mindistsgrid=2, maxdistsgrid=15,
                    dist_to_take=30., sigma_map=2.0, maxd_map = .3, n_fs = 7, n_intp = 7*6, nu=0.5, E=1, reg_lamb=2e-5, plat_scale=15, returning=False):
    """Pipeline to compute the traction forces on the test sporozoite."""
    mean_speed = (np.linalg.norm(biocent_vel1_sm[identified_pairs[:,0][testsporo-1]], axis=-1) + np.linalg.norm(biocent_vel2_sm[identified_pairs[:,1][testsporo-1]], axis=-1))/2
    sinks = signal.find_peaks(-mean_speed, distance=5, height=-5)[0]
    frames_to_take = np.zeros(len(frames_1)).astype(bool)
    frames_to_take[sinks] = True
    frames_to_take = ndimage.binary_dilation(frames_to_take, iterations=2)
    coords_map1, disps_map1, disps_map_av1, valid_map1, mean_f_c_angle1, mean_opening_angle1, mean_rad1, rads_here1, ang_here1, disp_ang1, disp_mag1 = single_sporo_pc_analysis(testsporo, identified_pairs[:,0], biocents1_ordered, disp1_sm, beadmean1, dist_to_take=dist_to_take, sigma_map=sigma_map, maxd_map=maxd_map, frames_to_take=frames_to_take)
    coords_map2, disps_map2, disps_map_av2, valid_map2, mean_f_c_angle2, mean_opening_angle2, mean_rad2, rads_here2, ang_here2, disp_ang2, disp_mag2 = single_sporo_pc_analysis(testsporo, identified_pairs[:,1], biocents2_ordered, disp2_sm, beadmean2, dist_to_take=dist_to_take, sigma_map=sigma_map, maxd_map=maxd_map, frames_to_take=frames_to_take)
    
    sporoangles1 = np.linspace(mean_f_c_angle1, mean_f_c_angle1-mean_opening_angle1*np.sign(mean_f_c_angle1), n_fs)
    print('Sporolength:', mean_opening_angle1*mean_rad1, ' Len per f:', mean_opening_angle1*mean_rad1/n_fs)
    f_coords1 = np.array([np.array([np.cos(a), -np.sin(a)]) for a in sporoangles1])*mean_rad1#*radmod[:,None]

    f_vecs1 = np.array([np.array([np.sin(a), np.cos(a)]) for a in sporoangles1])*(sporoangles1-sporoangles1.mean())[:,None]
    f_dips1 = 0.*np.ones_like(sporoangles1) 
    sporoangles2 = np.linspace(mean_f_c_angle2, mean_f_c_angle2-mean_opening_angle2*np.sign(mean_f_c_angle2), n_fs)

    f_coords2 = np.array([np.array([np.cos(a), -np.sin(a)]) for a in sporoangles2])*mean_rad2#*radmod[:,None]
    f_vecs2 = np.array([np.array([np.sin(a), np.cos(a)]) for a in sporoangles2])*(sporoangles2-sporoangles2.mean())[:,None]
    f_dips2 = 0.*np.ones_like(sporoangles2) 

    params_ini = {'f1': f_vecs1, 'f2': f_vecs2, 'd1': f_dips1, 'd2': f_dips2, 'plat_scale_d1': plat_scale[1], 'plat_scale_d2': plat_scale[1]}


    grid = np.stack(np.meshgrid(np.arange(coords_map1[:,0].min(), coords_map1[:,0].max(), gridres), np.arange(coords_map1[:,1].min(), coords_map1[:,1].max(), gridres)), axis=-1)

    coordsall = np.concatenate((coords_map1, coords_map2))
    gridx = np.arange(coordsall[:,0].min(), coordsall[:,0].max(), gridres)
    gridy = np.arange(coordsall[:,1].min(), coordsall[:,1].max(), gridres)
    grid = np.stack(np.meshgrid(gridx, gridy), axis=-1).reshape(-1,2)
    grid1, grid2 = grid.copy(), grid.copy()
    minmaxcheck = lambda x,min,max: (x>min)*(x<max)
    grid1[~minmaxcheck(np.linalg.norm(grid1[None]-f_coords1[:,None], axis=-1).min(axis=0), mindistsgrid, maxdistsgrid)] = np.nan
    grid2[~minmaxcheck(np.linalg.norm(grid2[None]-f_coords2[:,None], axis=-1).min(axis=0), mindistsgrid, maxdistsgrid)] = np.nan
    grid1 = grid1[~np.isnan(grid1).any(axis=-1)]
    grid2 = grid2[~np.isnan(grid2).any(axis=-1)]
    weights_grid1 = np.linalg.norm(grid1[None] - coords_map1[:,None], axis=-1)
    weights_grid1 = np.nan_to_num(np.exp(-weights_grid1**2/(2*sigmagrid**2)))
    weights_grid1 = weights_grid1/weights_grid1.sum(axis=0)[None]
    disps_grid1 = np.nansum((disps_map1/np.clip(np.linalg.norm(disps_map1, axis=-1)/maxdispgrid,1.,np.inf)[:,None])[:,None]*weights_grid1[:,:,None], axis=0)
    weights_grid2 = np.linalg.norm(grid2[None] - coords_map2[:,None], axis=-1)
    weights_grid2 = np.nan_to_num(np.exp(-weights_grid2**2/(2*sigmagrid**2)))
    weights_grid2 = weights_grid2/weights_grid2.sum(axis=0)[None]
    disps_grid2 = np.nansum((disps_map2/np.clip(np.linalg.norm(disps_map2, axis=-1)/maxdispgrid,1.,np.inf)[:,None])[:,None]*weights_grid2[:,:,None], axis=0)

    minimizee_h = jax.jit(lambda f_vecs_h: minimizee_sw(f_vecs_h, 
                                                                    grid1, disps_grid1, grid2, disps_grid2, f_coords1, f_coords2, nu, E, reg_lamb, plat_scale))

    params_flat, unravel = ravel_pytree(params_ini)
    def flat_minimizee_h(params_flat): #use flat array instdad of the PyTree Dict params
        params = unravel(params_flat)
        return minimizee_h(params)

    grad_fn = jax.jit(jax.grad(flat_minimizee_h))
    hess_fn = jax.jit(jax.hessian(flat_minimizee_h))

    optres = optimize.minimize(flat_minimizee_h, params_flat, method='trust-constr', jac=grad_fn, hess=hess_fn, options={'maxiter': 10000, 'disp': True, 'gtol': 1e-12})
    params_res = unravel(optres.x)
    print(optres.fun)

    nu_res = nu
    f_vecs_res1, f_dips_res1 = params_res['f1'], params_res['d1']
    f_vecs_res2, f_dips_res2 = params_res['f2'], params_res['d2']
    u_res1 = u_total(grid1, f_coords2, f_vecs_res1, f_dips_res1, nu_res, E, (plat_scale[0], params_res['plat_scale_d1']))
    u_res2 = u_total(grid2, f_coords2, f_vecs_res2, f_dips_res2, nu_res, E, (plat_scale[0], params_res['plat_scale_d2']))
    f_dips_anc_eigvals1, f_dips_anc_eigvecs1 = np.linalg.eig(build_f_dip_matrix(f_dips_res1, f_coords1))
    f_dips_anc_eigvals2, f_dips_anc_eigvecs2 = np.linalg.eig(build_f_dip_matrix(f_dips_res2, f_coords2))
    # Sort eigenvecs by eigenvals so eigvec1 is the one with the larger eigenval
    f_dips_anc_eigvecs1 = f_dips_anc_eigvecs1[np.arange(f_dips_anc_eigvals1.shape[0])[:,None,None], np.arange(2)[None,:,None], np.argsort(np.abs(f_dips_anc_eigvals1), axis=-1)[:,None]]
    f_dips_anc_eigvals1 = f_dips_anc_eigvals1[np.arange(f_dips_anc_eigvals1.shape[0])[:,None], np.argsort(np.abs(f_dips_anc_eigvals1))]
    f_dips_anc_eigvecs2 = f_dips_anc_eigvecs2[np.arange(f_dips_anc_eigvals2.shape[0])[:,None,None], np.arange(2)[None,:,None], np.argsort(np.abs(f_dips_anc_eigvals2), axis=-1)[:,None]]
    f_dips_anc_eigvals2 = f_dips_anc_eigvals2[np.arange(f_dips_anc_eigvals2.shape[0])[:,None], np.argsort(np.abs(f_dips_anc_eigvals2))]

    mosaic = [[0,1,2,4],[0,1,3,5]]
    dir_sign = np.sign(mean_f_c_angle1+mean_f_c_angle2)
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(24,10), width_ratios=[8,8,4,4])
    f_Scale, disp_scale = 0.5, .02
    ax[0].quiver(f_coords1[:,0], f_coords1[:,1], f_vecs_res1[:,0], f_vecs_res1[:,1], color='C0', zorder=3, scale=f_Scale/20, angles='xy', scale_units='xy')
    ev_tt = 1
    ev_cols1 = [viridis(c) for c in np.real((f_dips_anc_eigvals1[:,ev_tt]-f_dips_anc_eigvals1[:,ev_tt].min())/(f_dips_anc_eigvals1[:,ev_tt].max()-f_dips_anc_eigvals1[:,ev_tt].min()))]
    ax[0].quiver(f_coords1[:,0], f_coords1[:,1], f_dips_anc_eigvecs1[:,0,ev_tt], f_dips_anc_eigvecs1[:,1,ev_tt], color=ev_cols1, zorder=3, scale=f_Scale, angles='xy', scale_units='xy')
    ax[0].quiver(f_coords1[:,0], f_coords1[:,1], -f_dips_anc_eigvecs1[:,0,ev_tt], -f_dips_anc_eigvecs1[:,1,ev_tt], color=ev_cols1, zorder=3, scale=f_Scale, angles='xy', scale_units='xy')
    ax[0].quiver(grid1[:,0], grid1[:,1], u_res1[:,0], u_res1[:,1], color='r', zorder=0, scale=disp_scale, angles='xy', scale_units='xy')
    ax[0].quiver(grid1[:,0], grid1[:,1], disps_grid1[:,0], disps_grid1[:,1], color='g', zorder=-1, scale=disp_scale, angles='xy', scale_units='xy')
    ax[0].set_aspect('equal')
    ax[0].set_title('Top Gel')
    ax[1].quiver(f_coords2[:,0], f_coords2[:,1], f_vecs_res2[:,0], f_vecs_res2[:,1], color='C0', zorder=3, label='Force', scale=f_Scale/20, angles='xy', scale_units='xy')
    ev_cols2 = [viridis(c) for c in np.real((f_dips_anc_eigvals2[:,ev_tt]-f_dips_anc_eigvals2[:,ev_tt].min())/(f_dips_anc_eigvals2[:,ev_tt].max()-f_dips_anc_eigvals2[:,ev_tt].min()))]
    ax[1].quiver(f_coords2[:,0], f_coords2[:,1], f_dips_anc_eigvecs2[:,0,ev_tt], f_dips_anc_eigvecs2[:,1,ev_tt], color=ev_cols2, zorder=3, scale=f_Scale, angles='xy', scale_units='xy')
    ax[1].quiver(f_coords2[:,0], f_coords2[:,1], -f_dips_anc_eigvecs2[:,0,ev_tt], -f_dips_anc_eigvecs2[:,1,ev_tt], color=ev_cols2, zorder=3, scale=f_Scale, angles='xy', scale_units='xy')
    ax[1].quiver(grid2[:,0], grid2[:,1], u_res2[:,0], u_res2[:,1], color='r', zorder=0, scale=disp_scale, label='Reconstructed Dis.', angles='xy', scale_units='xy')
    ax[1].quiver(grid2[:,0], grid2[:,1], disps_grid2[:,0], disps_grid2[:,1], color='g', zorder=-1, scale=disp_scale, label='Measured Disp.', angles='xy', scale_units='xy')
    ax[1].set_title('Bottom Gel')
    ax[1].set_aspect('equal')

    for i, mean_f_c_angle, mean_opening_angle, mean_rad in zip([0,1], [mean_f_c_angle1, mean_f_c_angle2], [mean_opening_angle1, mean_opening_angle2], [mean_rad1, mean_rad2]):
        angles = (-mean_f_c_angle*180/np.pi, (np.sign(mean_f_c_angle)*mean_opening_angle-mean_f_c_angle)*180/np.pi)
        if mean_f_c_angle<0:
            angles = angles[::-1]
        sporoarc = Arc((0,0),2*mean_rad, 2*mean_rad, theta1=angles[0], theta2=angles[1], 
                        color=viridis(1.), zorder=2, lw=30, alpha=0.7, capstyle='round')
        ax[i].add_patch(sporoarc)
        front = np.array([mean_rad*np.cos(mean_f_c_angle), -mean_rad*np.sin(mean_f_c_angle)])
        ax[i].scatter(front[0], front[1], c='b', s=200, label='Front')

    ax[0].quiver(0,0,*(f_vecs_res1.sum(axis=0)), scale=f_Scale/20, angles='xy', scale_units='xy')
    ax[1].quiver(0,0,*(f_vecs_res2.sum(axis=0)), scale=f_Scale/20, angles='xy', scale_units='xy')
    ax[0].set_xlim((-10,40))
    ax[0].set_ylim((35,-35))
    ax[1].set_xlim((-10,40))
    ax[1].set_ylim((35,-35))
    ax[1].legend(loc='upper right')


    tangents1 = np.array([np.array([np.sin(a), np.cos(a)]) for a in sporoangles1])
    tang_fs1 = np.sum(f_vecs_res1*tangents1, axis=-1)
    tangents2 = np.array([np.array([np.sin(a), np.cos(a)]) for a in sporoangles2])
    tang_fs2 = np.sum(f_vecs_res2*tangents2, axis=-1)
    ax[2].set_title(f'Sum Top {dir_sign*tang_fs1.sum(axis=0):.2f}, Bottom {dir_sign*tang_fs2.sum(axis=0):.2f}')
    ax[2].plot(dir_sign*sporoangles1, dir_sign*tang_fs1, c='r', label='Tangetial Force Top')
    ax[2].plot(dir_sign*sporoangles2, dir_sign*tang_fs2, c='g', label='Tangetial Force Bottom')
    ax[2].legend()
    ax[2].set_xlabel('Angle (>0 is front)')


    ax[3].imshow(frames_1.max(axis=0)+frames_2.max(axis=0), cmap='bone')
    sporomask = (relabeled_both[:,0]==testsporo).any(axis=0)
    box = [(s.start-20, s.stop+20) for s in  ndimage.find_objects((relabeled_both[:,0]==testsporo).any(axis=0))[0]]
    im_s = ax[3].imshow(np.sqrt((relabeled_both[:,0]==testsporo).sum(axis=0)), cmap='viridis', interpolation='none', alpha=0.7*sporomask)
    ax[3].set_ylim(box[0][::-1])
    ax[3].set_xlim(box[1])
    ax[3].text(0.1,0.9,'CW Score: {:.2f}'.format(cwscore[testsporo-1]), transform=ax[3].transAxes, color='w')

    u_testangles1 = np.linspace(mean_f_c_angle1*1.05, 1.05*(mean_f_c_angle1-mean_opening_angle1*np.sign(mean_f_c_angle1)), n_fs+1)
    u_testcoords1 = np.array([[np.cos(a), -np.sin(a)] for a in u_testangles1])*mean_rad1
    u_testtangents1 = -np.array([[-np.sin(a), -np.cos(a)] for a in u_testangles1])
    u_sporo1 = u_total(u_testcoords1, f_coords1, f_vecs_res1, f_dips_res1, nu_res, E, plat_scale)
    u_testangles2 = np.linspace(mean_f_c_angle2*1.05, 1.05*(mean_f_c_angle2-mean_opening_angle2*np.sign(mean_f_c_angle2)), n_fs+1)
    u_testcoords2 = np.array([[np.cos(a), -np.sin(a)] for a in u_testangles2])*mean_rad2
    u_testtangents2 = -np.array([[-np.sin(a), -np.cos(a)] for a in u_testangles2])
    u_sporo2 = u_total(u_testcoords2, f_coords2, f_vecs_res2, f_dips_res2, nu_res, E, plat_scale)
    u_sporo_tang1 = np.einsum('ij,ij->i', u_sporo1, u_testtangents1)
    u_sporo_tang2 = np.einsum('ij,ij->i', u_sporo2, u_testtangents2)
    ax[4].plot(dir_sign*u_testangles1, dir_sign*u_sporo_tang1, c='r', label='Top')
    ax[4].plot(dir_sign*u_testangles2, dir_sign*u_sporo_tang2, c='g', label='Bottom')
    ax[4].set_title('Tan. Disp, Sum Top {:.2f}, Bottom {:.2f}'.format(dir_sign*u_sporo_tang1.sum(), dir_sign*u_sporo_tang2.sum()))
    ax[4].set_xlabel('Angle (>0 is front)')

    for d,ls,c,la,sa in zip([f_dips_anc_eigvals1[:,0],f_dips_anc_eigvals1[:,1],f_dips_anc_eigvals2[:,0],f_dips_anc_eigvals2[:,1]],['--','-','--','-'],['r','r','g','g'],['T_low','T_up','B_low','B_up'],[sporoangles1,sporoangles1,sporoangles2,sporoangles2]):
        ax[5].plot(dir_sign*sa, d, ls, c=c, label=la)
    ax[5].legend()
    ax[5].set_title('Orthogonal contraction')
    ax[5].set_xlabel('Angle (>0 is front)')
    if not plotname is None:
        plt.savefig(plotname, dpi=500, bbox_inches='tight')
    if not savename is None:
        resultdict = {'f_vecs_res1':f_vecs_res1, 'f_dips_res1':f_dips_res1, 'f_vecs_res2':f_vecs_res2, 'f_dips_res2':f_dips_res2, 'nu_res':nu_res,
                      'f_coords1':f_coords1, 'f_coords2':f_coords2,
                       'tang_fs1':tang_fs1, 'tang_fs2':tang_fs2, 'sum_tang_fs1':dir_sign*tang_fs1.sum(axis=0), 
                       'sum_tang_fs2':dir_sign*tang_fs2.sum(axis=0), 'cwscore':cwscore[testsporo-1], 'optres':None,
                       'u_t_under_sporo1':u_sporo_tang1, 'u_t_under_sporo2':u_sporo_tang2, 'u_under_sporo1':u_sporo1, 'u_under_sporo2':u_sporo2,
                       'dip_eigvals1':f_dips_anc_eigvals1, 'dip_eigvals2':f_dips_anc_eigvals2, 'dip_eigvecs1':f_dips_anc_eigvecs1, 'dip_eigvecs2':f_dips_anc_eigvecs2,
                       'coords_map1':coords_map1, 'disps_map1':disps_map1, 'disps_map_av1':disps_map_av1, 'valid_map1':valid_map1,
                       'coords_map2':coords_map2, 'disps_map2':disps_map2, 'disps_map_av2':disps_map_av2, 'valid_map2':valid_map2,
                       'mean_f_c_angle1':mean_f_c_angle1, 'mean_opening_angle1':mean_opening_angle1, 'mean_rad1':mean_rad1, 'rads_here1':rads_here1, 'ang_here1':ang_here1, 'disp_ang1':disp_ang1, 'disp_mag1':disp_mag1,
                       'mean_f_c_angle2':mean_f_c_angle2, 'mean_opening_angle2':mean_opening_angle2, 'mean_rad2':mean_rad2, 'rads_here2':rads_here2, 'ang_here2':ang_here2, 'disp_ang2':disp_ang2, 'disp_mag2':disp_mag2,
                       'grid1':grid1, 'disps_grid1':disps_grid1, 'grid2':grid2, 'disps_grid2':disps_grid2, 'u_res1':u_res1, 'u_res2':u_res2}
        np.save(savename, resultdict)
    plt.close()
    if returning:
        return resultdict
    

def pipeline_analysis_to_TF(filepath, filename, plotoutpath, saveoutpath):
    """Pipeline to compute the traction forces on all sporos identified in a previously analyzed ROI."""
    resdict = np.load(filepath, allow_pickle=True).item()
    frames_1, frames_2, sporo, beadmean1, beadmean2, disp1_sm, disp2_sm, traj1, traj2 = resdict['frames_1'], resdict['frames_2'], resdict['sporo'], resdict['beadmean1'], resdict['beadmean2'], resdict['disp1_sm'], resdict['disp2_sm'], resdict['traj1'], resdict['traj2']
    labeled1, labeled2, track1id, track2id, massmeans = resdict['labeled1'], resdict['labeled2'], resdict['track1id'], resdict['track2id'], resdict['massmeans']
    ints1 = frames_1[track1id['frame'].values.astype(int), np.round(track1id['y']).values.astype(int), np.round(track1id['x']).values.astype(int)]
    ints2 = frames_2[track2id['frame'].values.astype(int), np.round(track2id['y']).values.astype(int), np.round(track2id['x']).values.astype(int)]
    intgather = np.zeros((track1id['particle'].max()+1,2,len(frames_1)), dtype=float)
    intgather.fill(np.nan)
    for i in np.unique(track1id['particle']):
        intgather[i,0,track1id['frame'].values[track1id['particle'].values == i]]=ints1[track1id['particle'].values == i]
        intgather[i,1,track2id['frame'].values[track2id['particle'].values == i]]=ints2[track2id['particle'].values == i]
    f1larger = np.nansum(intgather[:,0] > intgather[:,1]+50, axis=1)
    f2larger = np.nansum(intgather[:,1] > intgather[:,0]+50, axis=1)
    selector = (f1larger-f2larger)/(np.logical_not(np.isnan(intgather).any(axis=1)).sum(axis=1))

    selector_crit = 0.5
    in_zlayer_1 = selector>selector_crit
    in_zlayer_2 = selector<-selector_crit
    traj1 = [np.array(track1id[['x','y', 'frame']][track1id['particle']==i]) for i in track1id['particle'].unique() if in_zlayer_1[i]] + [np.array(track2id[['x','y', 'frame']][track2id['particle']==i]) for i in track2id['particle'].unique() if in_zlayer_1[i]]
    traj1 = np.array([fill_traj(t,len(frames_1)) for t in traj1])
    traj2 = [np.array(track2id[['x','y', 'frame']][track2id['particle']==i]) for i in track2id['particle'].unique() if in_zlayer_2[i]] + [np.array(track1id[['x','y', 'frame']][track1id['particle']==i]) for i in track1id['particle'].unique() if in_zlayer_2[i]]
    traj2 = np.array([fill_traj(t,len(frames_2)) for t in traj2])
    beadmean1 = np.nanmean(traj1, axis=1)
    beadmean2 = np.nanmean(traj2, axis=1)

    median_size = 31
    beadmean1, disp1_sm = pc_update_bm_and_disp(beadmean1, traj1, labeled1, median_size = median_size)
    beadmean2, disp2_sm = pc_update_bm_and_disp(beadmean2, traj2, labeled2, median_size = median_size)

    identified_pairs = identify_paired_sporos(labeled1, labeled2)
    relabeled1, relabeled2 = np.zeros_like(labeled1), np.zeros_like(labeled2)
    for i, pair in enumerate(identified_pairs):
        relabeled1[labeled1 == pair[0]] = i+1
        relabeled2[labeled2 == pair[1]] = i+1
    relabeled_both = np.where(relabeled1>0, relabeled1, relabeled2)

    biocents1_ordered, biocent_vel1_sm = pc_get_biocents_and_vels(resdict['tresh_sporos1'], resdict['box_positions_sporos1'], resdict['metadata'])
    biocents2_ordered, biocent_vel2_sm = pc_get_biocents_and_vels(resdict['tresh_sporos2'], resdict['box_positions_sporos2'], resdict['metadata'])

    cwscore1 = clockwise_scors(biocents1_ordered[identified_pairs[:,0],:,0])
    cwscore2 = clockwise_scors(biocents2_ordered[identified_pairs[:,1],:,0])
    if ((np.abs(cwscore1-cwscore2)/np.abs(cwscore1+cwscore2)>0.3)*(np.abs(cwscore1+cwscore2)>0.5)).any():
        print('Clockwise scores differ by more than 50%', (np.abs(cwscore1-cwscore2)/np.abs(cwscore1+cwscore2)))
    cwscore = (cwscore1+cwscore2)/2

    dist_to_take, sigma_map, maxd_map, n_fs, n_intp, nu, E, reg_lamb, plat_scale = 30., 1.0, 0.3, 20, 20, 0.5, 1, (0*1e-7, 1e-3, 3e-3), (4.,9.)
    gridres, sigmagrid, maxdispgrid, mindistsgrid, maxdistsgrid = 2., 2., .3, 2, 10
    sporo_far_dists, max_far_dist = 0, 20
    
    os.makedirs(plotoutpath, exist_ok=True)
    os.makedirs(saveoutpath, exist_ok=True)
    tangsums1, tangsums2, cwscoressums = [],[],[]
    for testsporo in range(1,len(identified_pairs)+1):
        try:
            n_frames_obsv = np.sum((~np.isnan(biocents1_ordered[identified_pairs[testsporo-1,0]]).any(axis=(-2,-1)))*(~np.isnan(biocents2_ordered[identified_pairs[testsporo-1,1]]).any(axis=(-2,-1))))
            if (np.abs(cwscore[testsporo-1])<0.2) | (n_frames_obsv<67):
                continue
            savename_h = saveoutpath+filename+f"_S{resdict['metadata']['series']}_{testsporo}.npy"
            plotname_h = plotoutpath+filename+f"_S{resdict['metadata']['series']}_{testsporo}.png"
            result_dict = tf_on_testsporo(testsporo, identified_pairs, biocents1_ordered, disp1_sm, beadmean1, biocent_vel1_sm, biocents2_ordered, disp2_sm, beadmean2, biocent_vel2_sm,
                            frames_1, frames_2, relabeled_both, cwscore, plotname=plotname_h, savename=savename_h, sporo_far_dists=sporo_far_dists, max_far_dist=max_far_dist,
                            gridres=gridres, sigmagrid=sigmagrid, maxdispgrid=maxdispgrid, mindistsgrid=mindistsgrid, maxdistsgrid=maxdistsgrid,
                            dist_to_take=dist_to_take, sigma_map=sigma_map, maxd_map=maxd_map, n_fs=n_fs, n_intp=n_intp, nu=nu, E=E, reg_lamb=reg_lamb, plat_scale=plat_scale, returning=True)
            tangsums1.append(result_dict['sum_tang_fs1'])
            tangsums2.append(result_dict['sum_tang_fs2'])
            cwscoressums.append(result_dict['cwscore'])
        except Exception as e:
            print(f'Error in Sporo {testsporo}: {e}')

    return tangsums1, tangsums2, cwscoressums