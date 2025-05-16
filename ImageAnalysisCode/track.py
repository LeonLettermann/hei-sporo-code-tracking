# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import cv2
import scipy.ndimage as ndimage
import threading


def thresh_n_label(stack, ada_thresh_width = 19, ada_thresh_strength=-55, abs_thresh=165, min_size=30, max_size=5000, min_size_per_z=0, max_size_per_z=5000, co_mask=None, dilate=0):
    """Thresholds a given 4D image stack, and labels found features
    """
    threshstack = np.zeros(stack.shape, dtype=bool)
    for frame in range(stack.shape[0]):
        for z in range(stack.shape[1]):
            threshstack[frame,z] = cv2.adaptiveThreshold(stack[frame, z], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ada_thresh_width, ada_thresh_strength)
    threshstack=threshstack*(stack>abs_thresh)
    if not co_mask is None:
        threshstack = threshstack*(~co_mask)

    struct = ndimage.generate_binary_structure(4,1)
    struct[0] = False
    struct[2] = False
    if dilate>0:
        threshstack = ndimage.binary_dilation(threshstack, structure=struct, iterations=dilate)
    threshstack = ndimage.binary_closing(np.pad(threshstack,((0,0),(3,3),(0,0),(0,0)), mode='edge'), structure=struct, iterations=2)[:,3:-3]

    struct = ndimage.generate_binary_structure(3, 2)
    struct[0]=False
    struct[2]=False
    for frame in range(threshstack.shape[0]):
        label, numl = ndimage.label(threshstack[frame], structure=struct)
        labelsize = np.bincount(label.flat)
        threshstack[frame] = np.where(np.logical_or(labelsize[label]<min_size_per_z,labelsize[label]>max_size_per_z), np.zeros_like(threshstack[frame]), threshstack[frame])

    struct = ndimage.generate_binary_structure(4, 2)
    struct[0]=False
    struct[2]=False
    label, numl = ndimage.label(threshstack, structure=struct)
    for frame in range(label.shape[0]):
        labelsize = np.bincount(label[frame].flat)
        label[frame] = np.where(np.logical_or(labelsize[label[frame]]<min_size,labelsize[label[frame]]>max_size), np.zeros_like(label[0]), label[frame])
    return label



def build_split_function(stack):
    """Builds a function that splits a label depending on two masks.
    This function make sure the splitting procedure is conducted correctly for the dimension of the stack.
    """
    kernel = np.exp(-np.linspace(-9,9,19)**2/10)
    kernel=kernel/np.sum(kernel)
    kernel=np.einsum('i,j,k->ijk', kernel if stack.shape[1]>1 else np.ones(1), kernel, kernel)
    def split_label(mask1, mask2):
        # positiv: split to 1, negative: split to 2
        split_decide = ndimage.convolve((1.0*mask1-1*mask2),kernel,mode='constant')
        return split_decide>=0
    return split_label

def track_label_in_frame(labels,labelval,proc_Frame, labels_used_in_frame, split_label, minsize):
    """Tracks a label in the current frame and splits it if necessary.
    This function is called in a thread for each label that has more than one parent."""
    to_split = (labels[proc_Frame]==labelval)
    roi = ndimage.find_objects(to_split)[0]
    to_split = to_split[roi]
    candidates = list(set((labels[proc_Frame-1][roi][to_split]).flat))
    if 0 in candidates: candidates.remove(0)
    len_c = len(candidates)
    if len_c==1: print("This is wrong!")
    if len_c>1: 
        overlap = np.array([(labels[proc_Frame-1][roi][to_split]==c).sum() for c in candidates])
        overlap_sorted = np.sort(overlap)
        newlabels = []
        if overlap_sorted[-1]/overlap_sorted[-2]>5 and overlap_sorted[-2]<minsize:
            c = candidates[overlap.argmax()]
            if c not in labels_used_in_frame:
                labels_used_in_frame.append(c)
                labels[proc_Frame][roi][to_split]=c      
                newlabels = [c]

        else:
            c_split_1, c_split_2 = candidates[overlap.argsort()[-1]], candidates[overlap.argsort()[-2]]

            mask1 = labels[proc_Frame-1][roi]==c_split_1
            mask2 = labels[proc_Frame-1][roi]==c_split_2
            split_decide = split_label(mask1, mask2)

            if np.sum((labels[proc_Frame][roi]==labelval)*(split_decide))>=minsize:
                if c_split_1 in labels_used_in_frame:
                    c_split_1 = labels.max()+1
                labels_used_in_frame.append(c_split_1)
                labels[proc_Frame][roi][(labels[proc_Frame][roi]==labelval)*(split_decide)]=c_split_1
                newlabels.append(c_split_1)
            else:
                labels[proc_Frame][roi][(labels[proc_Frame][roi]==labelval)*(split_decide)]=0

            if np.sum((labels[proc_Frame][roi]==labelval)*(~split_decide))>=minsize:
                if c_split_2 in labels_used_in_frame:
                    c_split_2 = labels.max()+1
                labels_used_in_frame.append(c_split_2)
                labels[proc_Frame][roi][(labels[proc_Frame][roi]==labelval)*(~split_decide)]=c_split_2
                newlabels.append(c_split_2)
            else:
                labels[proc_Frame][roi][(labels[proc_Frame][roi]==labelval)*(~split_decide)]=0
        
        # check if splitting created label with two connected components:
        for l in newlabels:
            label_l, numl = ndimage.label(labels[proc_Frame][roi]==l, ndimage.generate_binary_structure(3, 2))
            if numl>1:
                label_lsize = np.bincount(label_l.flat)[1:]
                for k in range(numl):
                    if k == label_lsize.argmax():
                        continue
                    else:
                        if label_lsize[k]<minsize:
                            labels[proc_Frame][roi][label_l==(k+1)]=0
                        else:
                            labels_used_in_frame.append(labels.max()+1)
                            labels[proc_Frame][roi][label_l==(k+1)]=labels.max()+1

def track_one_direction(labels, minsize):
    """Track the labels through the stack in one direction."""
    split_label = build_split_function(labels)
    for proc_Frame in range(1,labels.shape[0]):
        label_at_frame = list(set(labels[proc_Frame].flat))
        if len(label_at_frame)==1 or labels[proc_Frame-1].max()==0:
            continue
        label_at_frame.remove(0)
        labels_used_in_frame = label_at_frame.copy()
        label_at_frame = np.array(label_at_frame)

        label_multiplicity = ndimage.labeled_comprehension(labels[proc_Frame-1], labels[proc_Frame], label_at_frame, lambda x: len(np.unique(x.flat))-(0 in x), int, 0)
        more_parents = label_at_frame[label_multiplicity>1] ## CHANGED!
        if 1 in label_multiplicity:
            #print(label_multiplicity)

            one_parent = label_at_frame[label_multiplicity==1]
            parent_indcs = ndimage.labeled_comprehension(labels[proc_Frame-1], labels[proc_Frame], one_parent, 
                                                         lambda x: x.max(), int, 0)
            parent_indcs_uni, invers, counter = np.unique(parent_indcs, return_inverse=True, return_counts=True)
            # find largest for not valid!
            largest_children_indcs = ndimage.labeled_comprehension(labels[proc_Frame], labels[proc_Frame-1], parent_indcs_uni, 
                                                         lambda x: np.bincount(x[x>0]).argmax(), int, 0)
            
            valid_one_parents = np.in1d(one_parent, largest_children_indcs)

            parent_matrix = np.zeros(label_at_frame.max()+1, dtype=np.int32)
            parent_matrix[one_parent[valid_one_parents]] = parent_indcs[valid_one_parents]
            parent_matrix = parent_matrix[labels[proc_Frame]]
            labels[proc_Frame] = np.where(parent_matrix>0, parent_matrix, labels[proc_Frame])
            labels_used_in_frame = labels_used_in_frame+list(parent_indcs[valid_one_parents])

        threads = []
        for labelval in list(more_parents):#+one_parent_labelvals_unused:
            threads.append(threading.Thread(target = (lambda lv: track_label_in_frame(labels,lv,proc_Frame, labels_used_in_frame, split_label, minsize)),
                                                    args=(labelval,)))
            threads[-1].start()
        for t in threads: t.join()


def enforce_connected_tracks(labeled):
    """Enforces that all labels are connected in the stack, i.e. no label vanishes and then reappears."""
    i = 1
    max_label = labeled.max()
    roi = ndimage.find_objects(labeled)
    added = 0
    while i <= max_label:
        if roi[i-1] is None:
            i+=1
            continue

        newlabel, numl = ndimage.label(labeled[roi[i-1]]==i, ndimage.generate_binary_structure(4, 2))
        if numl>1:
            #print(i, numl)
            newlabelsize = np.bincount(newlabel.flat)[1:]
            for k in range(numl):
                if k == newlabelsize.argmax():
                    continue
                else:
                    max_label+=1
                    labeled[roi[i-1]][newlabel==(k+1)]=max_label
                    roi=roi+ndimage.find_objects(labeled==max_label)
                    added += 1
        i+=1
    print('{} Labels Added'.format(added))

def gen_occ(i, label):
    occ = np.zeros(label.max()+1).astype(bool)
    occ[np.unique(label[i])] = True
    return occ

def reconn_tracks(label, metadata, max_dist = 10):
    """Reconnects tracks that are not connected in the stack.
    This is done by finding the center of mass of the label in the two frames and connecting them if they are close enough.
    """
    pxlsizes = np.array([metadata['SizeZ'], metadata['SizeX'], metadata['SizeY']])
    for i in range(label.shape[0]-1):
        occ_i, occ_i1 = gen_occ(i, label), gen_occ(i+1, label)
        end_at_i = np.where(occ_i*np.logical_not(occ_i1))[0]
        begin_after_i = np.where(np.logical_not(occ_i)*occ_i1)[0]
        if len(end_at_i)==0 or len(begin_after_i)==0:
            continue
        try:
            coms_at_i = np.array(ndimage.center_of_mass(label[i]>0, label[i], end_at_i))*pxlsizes[None]
            coms_after_i = np.array(ndimage.center_of_mass(label[i+1]>0, label[i+1], begin_after_i))*pxlsizes[None]
            dd = np.linalg.norm(coms_at_i[:,np.newaxis] - coms_after_i[np.newaxis], axis=-1)
            dd[dd>max_dist] = np.inf
            minindices = np.argmin(dd, axis=1)
            minindices[np.min(dd, axis=1)==np.inf] = -1
            dublicates = np.where(np.bincount(minindices[minindices>=0])>1)[0]
            for d in dublicates:
                origin_indices = np.where(minindices==d)[0]
                dubdists = dd[origin_indices, d]
                mindistamongdubs = dubdists.argmin()
                for k in range(len(origin_indices)):
                    if k!=mindistamongdubs:
                        minindices[origin_indices[k]]=-1

            nearest_candidate = begin_after_i[minindices[minindices>=0]]
            end_at_i = end_at_i[minindices>=0]
            #print(i, len(end_at_i), len(begin_after_i), len(nearest_candidate))
            label_exchanger = np.arange(label.max()+1)
            for j, n in enumerate(end_at_i):
                label_exchanger[nearest_candidate[j]] = n
            label = label_exchanger[label]
        except Exception as e:
            print(e)
            continue
    return label

def track(label, metadata, max_reconn_dist = 10, min_size = 30):
    """Tracks the label in the stack.
    This is done by tracking the label in forward direction and then in 
    the backward direction, repeating this 3 times.
    """
    label_backwards = np.flip(label, 0)
    for i in range(3):
        track_one_direction(label, min_size)
        print("Forward complete",i)
        track_one_direction(label_backwards, min_size)
        print("Backward complete", i)
    
    enforce_connected_tracks(label)

    if max_reconn_dist>0:
        label = reconn_tracks(label, metadata, max_reconn_dist)


    labelvals = np.unique(label.flat)
    labelvals.sort()
    if labelvals[0]==0:
        labelvals = labelvals[1:]
    time_appearance = [s[0].stop-s[0].start-1 for s in ndimage.find_objects(label) if not s is None]
    excluded=0
    new_labelvals = np.zeros(labelvals.max()+1, dtype=np.int32)
    for i in range(len(labelvals)):
        if time_appearance[i]<10:
            excluded+=1
        else:
            new_labelvals[labelvals[i]]=i+1-excluded
    label = new_labelvals[label]
    print(i+1, excluded)

    return label.astype(np.uint16)
