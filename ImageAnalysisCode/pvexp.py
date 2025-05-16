# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import numpy as np
import pyevtk
import matplotlib.pyplot as plt
ac = np.ascontiguousarray

def export_tracks(positions, output, extradata = None):
    """
    Exports tracks to paraview readable file.
    """
    x,y,z,pointsPerLine, lvs, substclose, longer, in_roi = [],[],[],[],[], [], [], []
    if not extradata is None:
        extradatalists = {}
        for k in extradata.keys():
            extradatalists[k] = []
    for lv in range(positions.shape[0]):
        c = 0
        for frame in range(positions.shape[1]):
            p =positions[lv, frame]
            if not np.isnan(p[0]):
                x.append(p[0])
                y.append(p[1])
                z.append(p[2])
                c+=1
                lvs.append(lv)
                if not extradata is None:
                    for k in extradata.keys():
                        extradatalists[k].append(extradata[k][lv,frame])
        norms = np.linalg.norm(np.diff(positions[lv], axis=0), axis=1)
        longer.extend(c*[1]) if np.nansum(norms)>50 else longer.extend(c*[0])

        pointsPerLine.append(c)

    pointData = {'Labels':ac(lvs), 'Longer':ac(longer)}
    if not extradata is None:
        for k in extradata.keys():
            pointData[k] = ac(extradatalists[k])

    pyevtk.hl.polyLinesToVTK(output, ac(x), ac(y), ac(z), ac(pointsPerLine), pointData=pointData)

def export_stillframe(frame, output):
    """ Exports a still frame to paraview readable file."""
    pyevtk.hl.imageToVTK(output, cellData={"occupation": ac(frame)})

def export_pointsets(biocents, boxpos, skinned, metadata, outpath):
    """
    Exports pointsets to paraview readable file.
    """
    for t in range(len(biocents[0])):
        x,y,z,l = [],[],[],[]
        for i in range(len(biocents)):
            pos = (boxpos[i,t][np.newaxis]+np.stack(np.where(skinned[i][t]), axis=-1))*np.array([metadata['SizeZ'],metadata['SizeX'], metadata['SizeY']])[np.newaxis]
            x.extend(pos[:,1])
            y.extend(pos[:,2])
            z.extend(pos[:,0])
            l.extend(np.ones(len(pos))*i)
        pyevtk.hl.pointsToVTK(outpath + 'Skinned_{:03d}'.format(t), np.array(x), np.array(y), np.array(z), data={'Label':np.array(l)})