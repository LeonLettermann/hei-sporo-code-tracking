# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def plot_tracks(path, show=False, savepath=None, ax = None, fig=None):
    """
    Plots the tracks of a given file. The file should contain a 3D array of shape (n_tracks, n_frames, 3) with the positions of the tracks.
    The function will plot the tracks in 3D and save the figure if a savepath is provided.
    """
    com_skinned = np.load(path)
    steps = np.diff(com_skinned, axis=1)
    norms = np.linalg.norm(steps, axis=-1)
    total_length = np.nansum(norms, axis=-1)
    max_norms = np.nanmax(norms, axis=1)
    ptp = np.linalg.norm(np.nanmax(com_skinned, axis=1)-np.nanmin(com_skinned, axis=1), axis=1)
    valid = (ptp>5)*(max_norms<10)
    
    figreturn=False 
    if ax is None:
        figreturn=True 
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(projection='3d')
    
    if valid.sum()==0:
        print('Nothing!', path)
        valid = np.ones_like(valid)
        ax.set_title('No valid tracks!')
    ok = total_length[valid].argsort()[-np.minimum(50, valid.sum()):]

    positions = com_skinned[valid][ok]
    for i in range(positions.shape[0]):
        ax.plot(positions[i,:,0], positions[i,:,1], positions[i,:,2])
    ax.invert_zaxis()
    ax.set_box_aspect([np.ptp(positions[:,:,i][~np.isnan(positions[:,:,0])]) for i in range(3)])
    ax.set_xlabel('Valid Tracks: {}, cum. length: {:.1f}'.format(len(positions), total_length[valid][ok].sum()))
    if not savepath is None:
        plt.savefig(savepath)
    elif show:
        plt.show()
    else:
        return fig
    plt.close()

def plot_tracks_f_array(positions, show=False, savepath=None, ax = None, fig=None):
    """
    Plots the tracks of an array that should contain a 3D array of shape (n_tracks, n_frames, 3) with the positions of the tracks."""
    set_aspect=False
    if ax is None:
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(projection='3d')
        set_aspect=True

    for i in range(positions.shape[0]):
        ax.plot(positions[i,:,0], positions[i,:,1], positions[i,:,2])
    ax.invert_zaxis()
    if set_aspect:
        ax.set_aspect('equal')
    if not savepath is None:
        plt.savefig(savepath)
    elif show:
        plt.show()
    else:
        plt.close()
        return fig


def plot_helixprops(path, show=False, savepath=None, lossvalcutoff=8):
    """
    Plots the helix properties of a given result file. The file should contain a dictionary with the keys 'params', 'lossval', and 'inverted'.
    """
    helixdata = np.load(path, allow_pickle=True).item()
    ps = helixdata['params'][:,:,0][(helixdata['lossval']<lossvalcutoff)]
    radii = np.abs(helixdata['params'][:,:,1][(helixdata['lossval']<lossvalcutoff)])
    inverted = helixdata['inverted'][(helixdata['lossval']<lossvalcutoff)]
    ps_corr = ps*(1-2*inverted)
    fig, ax = plt.subplots(1,3,figsize=(10,4))
    ax[0].hist(2*np.pi*ps_corr, bins=100, range=(-20,20));
    ax[0].set_xlabel('Pitch $p$ in $\mu m$')
    ax[1].hist(radii, bins=100, range=(0,10));
    ax[1].set_xlabel('Radius $r$ in $\mu m$')
    ax[2].scatter(2*np.pi*ps_corr, radii, marker='x', alpha=0.2, s=3)
    ax[2].set_xlim((-20,20))
    ax[2].set_xlabel('Pitch $p$ in $\mu m$')
    ax[2].set_ylim((0,10))
    ax[2].set_ylabel('Radius $r$ in $\mu m$')
    fig.suptitle('#Samples: {} in {} Tracks'.format(len(radii), helixdata['params'].shape[0]-1))
    plt.tight_layout()
    if show: plt.show()
    if not savepath is None:
        plt.savefig(savepath)
    plt.close()

def plot_detailed_mediane(path, show=False, savepath=None, lossvalcutoff=8, strict3D=False, longest=None):
    """
    Plots the median of the helix properties of a given result folder.
    This selects the valid trajectories based on the given criteria and plots the median of the helix properties.
    """
    name = 'PC'
    biocents = np.load(path + name + '_Biocents_units.npy', allow_pickle=True)
    metadata = np.load(path + name + '_Metadata.npy', allow_pickle=True).item()
    helixdata = np.load(path + name + '_Helixfits.npy', allow_pickle=True).item()
    skinned = np.load(path + name + '_Shapes_Skinned.npy', allow_pickle=True)
    framedur = metadata['Delta_T']

    steps = np.diff(biocents, axis=1)
    normsteps = np.linalg.norm(steps, axis=2)
    cumlength = np.nansum(normsteps, axis=1)
    meanvel = (cumlength/((~np.isnan(biocents[:,:,0])).sum(axis=1)-1))/framedur
    totdist = np.linalg.norm(np.nanmax(biocents, axis=1)-np.nanmin(biocents, axis=1), axis=1)
    n_frames = (~np.isnan(biocents[:,:,0])).sum(axis=1)

    helixvel = np.sqrt(np.sum(helixdata['params'][:,:,:2]**2, axis=-1))*np.abs(np.nan_to_num(helixdata['ts_opt'][:,:,5]-helixdata['ts_opt'][:,:,4])/framedur+np.nan_to_num(helixdata['ts_opt'][:,:,6]-helixdata['ts_opt'][:,:,5])/framedur)
    
    def pca_shape(shape):
        pos = np.stack(np.where(shape), axis=-1)*np.array([metadata['SizeZ'],metadata['SizeX'], metadata['SizeY']])[np.newaxis]
        cov = np.cov(pos.T-np.mean(pos, axis=0)[np.newaxis].T)
        eigval, eigvec = np.linalg.eig(cov)
        return eigval

    def valid_shape(shape):
        if shape is None:
            return False
        pcas = pca_shape(shape)
        return (np.sum(shape) > 10)*(np.mean(np.sort(pcas)[:2])*2.5<pcas.max())
    shapevalidity = np.array([[False]*biocents.shape[1]]+[[valid_shape(s) for s in track] for track in skinned if not track is None])
    percvalid = (np.sum(shapevalidity, axis=1)/(~np.isnan(biocents[:,:,1])).sum(axis=1))>0.92

    valid = (n_frames>15)*percvalid# 30  18  *((np.nanmax(biocents, axis=1)-np.nanmin(biocents, axis=1))>2).all(axis=1)

    if strict3D:
        edgedist = 3
        closeedge = (biocents[:,:,2]>((metadata['n_z']-1)*metadata['SizeZ']-edgedist)) | (biocents[:,:,2]<edgedist)
        valid3d = ((closeedge*(~np.isnan(biocents[:,:,2]))).sum(axis=1)/(~np.isnan(biocents[:,:,2])).sum(axis=1))<1/3
        valid = valid*valid3d*(totdist>15)*(meanvel>0.2)

    if not longest is None:
        totdist = totdist*percvalid
        valid = (totdist>=np.sort(totdist)[-longest if percvalid.sum()>longest else -percvalid.sum()])

    to_take = (helixdata['lossval'][valid]<lossvalcutoff)*(ndimage.convolve(1*~np.isnan(biocents[valid][:,:,0]), np.ones((1,11)), mode='constant', cval=0)>=7)#*(helixvel[valid]>0.2)
    to_few = to_take.sum(axis=1)<6
    valid[np.where(valid)[0][to_few]]=False
    to_take = to_take[~to_few]
    
    
    ps = [helixdata['params'][valid][i,:,0][to_take[i]] for i in range(valid.sum())]
    radii = [np.abs(helixdata['params'][valid][i,:,1][to_take[i]]) for i in range(valid.sum())]
    inverted = [helixdata['inverted'][valid][i][to_take[i]] for i in range(valid.sum())]
    ps_corr = [ps[i]*(1-2*inverted[i]) for i in range(valid.sum())]

    helixvel_valid = [helixvel[valid][i][(helixdata['lossval'][valid][i]<lossvalcutoff)*(ndimage.convolve(1*(helixvel[valid][i]>0.1), np.ones(5), mode='constant')>2)] for i in range(valid.sum())]
    
    median_ps_corr = np.array([np.nanmedian(p) for p in ps_corr])
    median_radii = np.array([np.nanmedian(p) for p in radii])
    median_helixvel = np.array([np.nanmedian(p) for p in helixvel_valid])

    spread = 0.3
    fig, ax = plt.subplot_mosaic([['a','a','b']], figsize=(12,4))
    ax['a'].grid()
    ax['a'].scatter(np.arange(valid.sum()), median_ps_corr, c='C0', label='Pitch/2$\pi$ ($\mu m$)', alpha=0.7, s=50)
    ax['a'].scatter(np.arange(valid.sum()), median_radii, c='C1', label='Radius ($\mu m$)', alpha=0.7, s=50)
    ax['a'].scatter(np.arange(valid.sum()), median_helixvel, c='C2', label='Speed ($\mu m/s$)', alpha=0.7, s=50)
    ax['b'].remove()
    ax['b'] = plt.subplot(1,3,3,projection='3d')
    for i in range(valid.sum()):
        ax['a'].scatter(i*np.ones(len(ps_corr[i]))+spread*2*(0.5-np.random.rand(len(ps_corr[i]))), ps_corr[i], c='C0', marker='x', s=10, alpha=0.5)
        ax['a'].scatter(i*np.ones(len(radii[i]))+spread*2*(0.5-np.random.rand(len(radii[i]))), radii[i], c='C1', marker='x', s=10, alpha=0.5)
        ax['a'].scatter(i*np.ones(len(helixvel_valid[i]))+spread*2*(0.5-np.random.rand(len(helixvel_valid[i]))), helixvel_valid[i], c='C2', marker='x', s=10, alpha=0.5)
        ax['b'].plot(*biocents[valid][i].T, label=i+1)

    ax['a'].legend(ncols=3)
    ax['a'].set_xticks(np.arange(valid.sum()))
    ax['a'].set_xticklabels(np.arange(valid.sum())+1)
    ax['a'].set_xlabel('Track Number')
    ax['a'].set_ylim((-2,7.5))

    ax['b'].set_aspect('equal')
    ax['b'].set_xticklabels([])
    ax['b'].set_yticklabels([])
    ax['b'].set_zticklabels([])
    if valid.sum()<15: ax['b'].legend(bbox_to_anchor=(1.3,1))

    outpath = path + name + '_Helixfits.npy'
    outpath[::-1].find('corP/')
    dataname = outpath[outpath.find('MirkoFebruary')+14:-outpath[::-1].find('corP/')-5]+' S' + outpath[-19:-17]

    fig.suptitle(dataname)

    if show: plt.show()
    if not savepath is None:
        plt.savefig(savepath + dataname + '.png')
    plt.close()
    return median_ps_corr, median_radii, median_helixvel, np.where(valid)[0], ps_corr, radii, helixvel_valid, biocents[valid]


def gather_helixprops(path, savepath, strict3D=False, sporomedian=True, longest=None, longest_per_series=True):
    """
    gathers helics properties of all results within the subdirectories of the given path.
    The results are saved in a single file and the median of the properties is plotted.
    """
    dataname = path[path.find('MirkoFebruary')+14:-path[::-1].find('corP/')-5]

    if not os.path.isdir(savepath+'Details/'+dataname):
        os.mkdir(savepath+'Details/'+dataname)

    list_ps, list_rad, list_vel, list_tracknumbers, serieses, biocents = [],[],[],[],[], []

    for s in sorted(os.listdir(path)):
        pathhere = path+s+'/'
        series = int(s)
        if os.path.isfile(pathhere+'PC_Helixfits.npy'):
            m_ps, m_rad, m_vel, tracknumbers, ps_corr, radii, helixvel_valid, biocents_valid = plot_detailed_mediane(pathhere, savepath=savepath+'Details/'+dataname+'/Series_{}.png'.format(series), strict3D=strict3D, longest=longest)
            if sporomedian:
                list_ps.append(2*np.pi*m_ps)
                list_rad.append(m_rad)
                list_vel.append(m_vel)
            else:
                list_ps.append(np.concatenate([2*np.pi*p[~np.isnan(p)] for p in ps_corr]) if len(ps_corr)>0 else [])
                list_rad.append(np.concatenate([r[~np.isnan(r)] for r in radii]) if len(ps_corr)>0 else [])
                list_vel.append(np.concatenate([h[~np.isnan(h)] for h in helixvel_valid]) if len(ps_corr)>0 else [])
            list_tracknumbers.append(tracknumbers)
            serieses.append(series)
            biocents.append(biocents_valid)

    if not longest is None and longest_per_series:
        totdist = [np.linalg.norm(np.nanmax(biocents[i], axis=1)-np.nanmin(biocents[i], axis=1), axis=-1) for i in range(len(biocents))]
        cutoff = np.sort(np.concatenate(totdist))[-longest if len(np.concatenate(totdist))>longest else -len(np.concatenate(totdist))]
        to_take = [totdist[i]>=cutoff for i in range(len(totdist))]
        list_ps = [list_ps[i][to_take[i]] for i in range(len(list_ps))]
        list_rad = [list_rad[i][to_take[i]] for i in range(len(list_rad))]
        list_vel = [list_vel[i][to_take[i]] for i in range(len(list_vel))]
        list_tracknumbers = [list_tracknumbers[i][to_take[i]] for i in range(len(list_tracknumbers))]
        biocents = [biocents[i][to_take[i]] for i in range(len(biocents))]
        fig, ax = plt.subplot_mosaic([[0,1,2],[3,3,3],[3,3,3],[3,3,3]], figsize=(12,15), subplot_kw={'projection':'3d'})
        for i in range(3):
            ax[i].remove()
            ax[i]=fig.add_subplot(4,3,i+1)
        for i in range(len(biocents)):
            for j in range(len(biocents[i])):
                ax[3].plot(biocents[i][j,:,0], biocents[i][j,:,1], biocents[i][j,:,2])
        ax[3].invert_zaxis()
        ax[3].set_aspect('equal')

    else:
        fig, ax = plt.subplots(1,3,figsize=(10,4))

    np.save(savepath+'Data/'+dataname+'.npy', {'ps':list_ps, 'rad':list_rad, 'vel':list_vel, 'tracknumber':list_tracknumbers, 'series':serieses, 'biocents':biocents})

    for i in range(3):
        ax[i].set_xticklabels(['All']+serieses);
        ax[i].set_xlabel('Series');

    ax[0].boxplot([np.concatenate(list_ps)]+list_ps);
    ax[0].set_ylabel('Pitch ($\mu m$)')

    ax[1].boxplot([np.concatenate(list_rad)]+list_rad);
    ax[1].set_ylabel('Radius ($\mu m$)')

    ax[2].boxplot([np.concatenate(list_vel)]+list_vel);
    ax[2].set_ylabel('Velocity ($\mu m/s$)')

    fig.suptitle(dataname+';  {} Sporos tracked in total'.format(len(np.concatenate(list_ps))))

    plt.tight_layout()
    plt.savefig(savepath+'Gathered/'+dataname+'.png')
    plt.close()