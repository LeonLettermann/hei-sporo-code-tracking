# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Leon Lettermann

import aicsimageio as ai
import numpy as np


def find_number(xml, descriptor, return_start=False):
    """Find a numerical value labeld by a descriptor in an xml file."""
    desc = descriptor+"=\""
    start = xml.find(desc)
    if start == -1:
        raise Warning("Descriptor not found!")
    else:
        start = start+ len(desc)
    end = start+xml[start:].find("\"")
    if return_start:
        return xml[start:end], start
    else:
        return xml[start:end]

def load_bioformat(path, series, ThreeD=True, ReadExpTimeOfCnl=None, return_xml=False):
    """
    loads microscopy data (if supported by bioformats) and returns the data and metadata
    path: str, series: starting at 0
    """
    data = ai.readers.bioformats_reader.BioFile(path, series=series)
    xml = data.ome_xml

    start_ind = xml.find('Image ID=\"Image:{}\"'.format(series))
    if xml[start_ind+3:].find('Image ID=\"Image:{}\"'.format(series))!=-1 or start_ind==-1:
        raise(Warning('Image:Series not found or found multiple times!'))
    end_ind = xml.find('Image ID=\"Image:{}\"'.format(series+1))

    rel_xml = xml[start_ind:end_ind]
    metadata = {'SizeX':float(find_number(rel_xml, 'PhysicalSizeX')),
                'SizeY':float(find_number(rel_xml, 'PhysicalSizeY')),
                'SizeZ':(float(find_number(rel_xml, 'PhysicalSizeZ')) if ThreeD else None),
                'Bits':float(find_number(rel_xml, 'SignificantBits')),
                'n_T': int(data.core_meta.shape[0]),
                'n_C': int(data.core_meta.shape[1]),
                'n_z': int(data.core_meta.shape[2]),
                'n_x': int(data.core_meta.shape[3]),
                'n_y': int(data.core_meta.shape[4]),
                'series': series}

    ind = 0
    times = []
    if not ReadExpTimeOfCnl is None:
        expTimes = [] if isinstance(ReadExpTimeOfCnl, int) else [[] for i in range(len(ReadExpTimeOfCnl))]
    while True:
        try:
            new_time, start = find_number(rel_xml[ind:], 'DeltaT', return_start=True)
            if not ReadExpTimeOfCnl is None:
                exp_time_here, start_exp_time = find_number(rel_xml[ind:], 'ExposureTime', return_start=True)
                if isinstance(ReadExpTimeOfCnl, int) and int(find_number(rel_xml[ind+start_exp_time:], 'TheC', return_start=False))==ReadExpTimeOfCnl:
                    expTimes.append(float(exp_time_here))
                elif isinstance(ReadExpTimeOfCnl, list):
                    for i in range(len(ReadExpTimeOfCnl)):
                        if int(find_number(rel_xml[ind+start_exp_time:], 'TheC', return_start=False))==ReadExpTimeOfCnl[i]:
                            expTimes[i].append(float(exp_time_here))
        except:
            break
        new_time=float(new_time)
        if len(times)==0 or new_time>times[-1]:
            times.append(new_time)
            ind += start+10
        elif new_time==times[-1]:
            ind += start+10
        else:
            break

    metadata['Delta_T']= np.mean(np.diff(times))
    metadata['Std_Delta_T'] = np.std(np.diff(times))
    if not ReadExpTimeOfCnl is None:
        if isinstance(ReadExpTimeOfCnl, int):
            metadata['ExposureTime'] = np.mean(expTimes)
            metadata['Std_ExposureTime'] = np.std(expTimes)
        else:
            metadata['ExposureTime'] = [np.mean(expTimes[i]) for i in range(len(expTimes))]
            metadata['Std_ExposureTime'] = [np.std(expTimes[i]) for i in range(len(expTimes))]
    if return_xml:
        return data, metadata, xml
    else:
        return data, metadata