#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:00:00 2022

@author: thoralfdietrich
this is the cleaned and (hopefully okayish) commented version of the plwave pyrocko beamforming script I used to scratch together things
# 28 is the line I already checked and cleaned (no guarantee this will always be updated)

Edited on Thu Dec 7 13:00:00 2023 by Gerolf Vent <gvent@uni-potsdam.de>
"""

#%%

import numpy as np
import warnings


#%% auxiliary functions (from https://github.com/fablindner/glseis/blob/master/array_analysis.py)

def calculate_CSDM(dft_array, neig=0, norm=True):
    """
    Calculate CSDM matrix for beamforming.
    :param dft_array: 2-Dim array containing DFTs of all stations
        and for multiple time windows. dim: [number stations, number windows]
    :param neig: Number of eigenvalues to project out.
    :param norm: If True, normalize CSDM matrix.
    """
    # CSDM matrix
    K = np.dot(dft_array, dft_array.conj().T)
    #if np.linalg.matrix_rank(K) < dft_array.shape[0]:
    #    warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

    # annul dominant source
    if neig > 0:
        K = annul_dominant_interferers(K, neig, dft_array)

    # normalize
    if norm:
        K /= np.linalg.norm(K)

    return K


def annul_dominant_interferers(CSDM, neig, data):
    """
    This routine cancels the strong interferers from the data by projecting the
    dominant eigenvectors of the cross-spectral-density matrix out of the data.
    :type CSDM: numpy.ndarray
    :param CSDM: cross-spectral-density matrix obtained from the data.
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
    :type data: numpy.ndarray
    :param data: the data which was used to calculate the CSDM. The projector is
        applied to it in order to cancel the strongest interferer.

    :return: numpy.ndarray
        csdm: the new cross-spectral-density matrix calculated from the data after
        the projector was applied to eliminate the strongest source.
    """

    # perform singular value decomposition to CSDM matrix
    u, s, vT = np.linalg.svd(CSDM)
    # chose only neig strongest eigenvectors
    u_m = u[:, :neig]   # columns are eigenvectors
    v_m = vT[:neig, :]  # rows (!) are eigenvectors
    # set-up projector
    proj = np.identity(CSDM.shape[0]) - np.dot(u_m, v_m)
    # apply projector to data - project largest eigenvectors out of data
    data = np.dot(proj, data)
    # calculate projected cross spectral density matrix
    csdm = np.dot(data, data.conj().T)
    return csdm


def phase_matching(replica, K, processor):
    """
    Do phase matching of the replica vector with the CSDM matrix.
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations (dim: [n_stats, n_param])
    :param K: 2-D array CSDM matrix (dim: [n_stats, n_stats])
    :param processor: Processor used for phase matching. bartlett or adaptive.
    """
    # calcualte inverse of CSDM matrix for adaptive processor
    if processor == "adaptive":
        K = np.linalg.inv(K)

    # reshape K matrix (or inverse of K) and append copy of it n_param times
    # along third dimension
    n_stats, n_param = replica.shape
    K = np.reshape(K, (n_stats, n_stats, 1))
    K = np.tile(K, (1, 1, n_param))

    # bartlett processor
    if processor == "bartlett":
        # initialize array for dot product
        dot1 = np.zeros((n_stats, n_param), dtype=complex)
        # first dot product - replica.conj().T with K
        for i in range(n_stats):
            dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
        # second dot product - dot1 with replica
        beam = abs(np.sum(np.multiply(dot1, replica), axis=0))

    # adaptive processor
    elif processor == "adaptive":
        # initialize array for dot product
        dot1 = np.zeros((n_stats, n_param), dtype=complex)
        # first dot product - replica.conj().T with K_inv
        for i in range(n_stats):
            dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
        # second dot product - dot1 with replica
        dot2 = np.sum(np.multiply(dot1, replica), axis=0)
        beam = abs((1. + 0.j) / dot2)

    return beam


def plwave_beamformer(data, scoord, svmin, svmax, dsv, slow, fmin, fmax, Fs, w_length,
        w_delay, baz=None, processor="bartlett", df=0.2, neig=0, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).

    :type data: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type svmin, svmax: float
    :param svmin, svmax: slowness/velocity interval used to calculate replica vector
    :type dsv: float
    :param dsv: slowness/velocity step used to calculate replica vector
    :type slow: boolean
    :param slow: if true, svmin, svmax, dsv are slowness values. if false, velocity values
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type baz: float
    :param baz: Back azimuth. If given, the beam is calculated only for this specific back azimuth
    :type processor: string
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vecotr. see Corciulo et al., 2012
    :type df: float
    :param df: frequency step between fmin and fmax
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
        enables to suppress strong sources.
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: three numpy arrays:
        teta: back azimuth (dim: [number of bazs, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number of bazs, number of cs])
    
    
    Parameters differing from matchedfield_beamformer fct:
    svmin, svmax, dsv, baz=None
    """

    # number of stations
    n_stats = data.shape[1]

    # grid for search over backazimuth and apparent velocity
    if baz is None:
        teta = np.arange(1, 363, 2) + 180
    else:
        teta = np.array([baz + 180])
    if slow:
        s = np.arange(svmin, svmax + dsv, dsv) / 1000.
    else:
        v = np.arange(svmin, svmax + dsv, dsv) * 1000.
        s = 1. / v

    # create meshgrids
    teta_, s_ = np.meshgrid(teta, s)
    n_param = teta_.size
    # reshape
    teta_ = teta_.reshape(n_param)
    s_ = s_.reshape(n_param)
    # reshape for efficient calculation
    xscoord = np.tile(scoord[:,0].reshape(n_stats, 1), (1, n_param))
    yscoord = np.tile(scoord[:,1].reshape(n_stats, 1), (1, n_param))
    teta_ = np.tile(teta_, (n_stats, 1))
    s_ = np.tile(s_, (n_stats, 1))

    # extract number of data points
    npts = data[:, 1].size
    # construct analysis frequencies
    freq = np.arange(fmin, fmax+df, df)
    # construct time vector for sliding window
    w_time = np.arange(0, w_length, 1./Fs)
    npts_win = w_time.size
    npts_delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    nshots = int(np.floor((npts - w_time.size) / npts_delay)) + 1

    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    # vect_data = np.zeros((freq.size, n_stats, nshots), dtype=np.complex) #np.complex is deprecated
    vect_data = np.zeros((freq.size, n_stats, nshots), dtype=complex)


    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(w_time[:, None], freq[:, None].T))

    # initialize beamformer
    # dim: [n_param]
    beamformer = np.zeros(n_param)

    # calculate DFTs
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        n = 0
        while (n * npts_delay + npts_win) <= npts:
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[n*npts_delay: n*npts_delay+npts_win][:,None],
                     np.ones((1, len(freq))))
            # mean averages over time axis
            data_freq = np.mean(np.multiply(adjust, matrice_int), axis=0)
            # fill data steering vector: ii'th station, n'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data[:, ii, n] = (data_freq / abs(data_freq)).conj().T
            n += 1

    # loop over frequencies and do phase matching
    for ll in range(len(freq)):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        K = calculate_CSDM(vect_data[ll,:,:], neig, norm)

        # calculate replica vector
        replica = np.exp(-1j * (xscoord * np.cos(np.radians(90 - teta_)) \
                              + yscoord * np.sin(np.radians(90 - teta_))) \
                              * 2. * np.pi * freq[ll] * s_)
        replica /= np.linalg.norm(replica, axis=0)
        replica = np.reshape(replica, (n_stats, n_param))

        # do phase matching
        beamformer += phase_matching(replica, K, processor)

    # normalize by deviding through number of discrete frequencies
    beamformer /= freq.size
    # reshape, dim: [number baz, number slowness]
    beamformer = np.reshape(beamformer, (s.size, teta.size))
    teta -= 180
    return teta, s*1000., beamformer


def matchedfield_beamformer(data, scoord, xrng, yrng, zrng, dx, dy, dz, svrng, ds,
        slow, fmin, fmax, Fs, w_length, w_delay,  processor="bartlett", df=0.2,
        neig=0, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).
    Can also be used to focus the beam to a certain coordinate, which must be
    specified with xmax, ymax, zmax. In this case, dx, dy, and dz need to be set
    to zero!
    
    :type data: numpy.ndarray
    :param data: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type xrng, yrng, zrng: tuple
    :param xrng, yrng, zrng: parameters for spatial grid search. Grid ranges
        from xrng[0] to xrng[1], yrng[0] to yrng[1], and zrng[0] to zrng[1].
    :type dx, dy, dz: float
    :param dx, dy, dz: grid resolution; increment from xrng[0] to xrng[1],
        yrng[0] to yrng[1], zrng[0] to zrng[1]
    :type svrng: tuple
    :param svrng: slowness interval used to calculate replica vector
    :type ds: float
    :param ds: slowness step used to calculate replica vector
    :type slow: boolean 
    :param slow: if true, svmin, svmax, dsv are slowness values. if false, velocity values
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type processor: string
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vecotr. see Corciulo et al., 2012
    :type df: float
    :param df: frequency step between fmin and fmax
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
        enables to suppress strong sources.
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: four numpy arrays:
        xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
        ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number y-grid points, number x-grid points, number cs])
    
    Parameters differing from plwave_beamformer fct: 
    xrng, yrng, zrng, dx, dy, dz, svrng, ds
    """

    # number of stations
    n_stats = data.shape[1]

    # grid for search over location
    # if beam is fixed to a coordinate in x, y, or z
    if yrng[0] == yrng[1]:
        ycoord = np.array([yrng[0]])
    # if beam is calculated for a regular grid
    else:
        ycoord = np.arange(yrng[0], yrng[1] + dy, dy)
    # same for x ... 
    if xrng[0] == xrng[1]:
        xcoord = np.array([xrng[0]])
    else:
        xcoord = np.arange(xrng[0], xrng[1] + dx, dx)
    # and for z 
    if zrng[0] == zrng[1]:
        zcoord = np.array([zrng[0]])
    else:
        zcoord = np.arange(zrng[0], zrng[1] + dz, dz)
    # create meshgrids
    ygrid, xgrid = np.meshgrid(ycoord, xcoord)
    zgrid = np.zeros(xgrid.shape)
    ygrid = ygrid.reshape(ygrid.size)
    xgrid = xgrid.reshape(xgrid.size)
    zgrid = zgrid.reshape(zgrid.size)
    if zcoord.size > 1:
        ygrid = np.tile(ygrid, zcoord.size)
        xgrid = np.tile(xgrid, zcoord.size)
        zgrid_ = np.copy(zgrid)
        for i in range(zcoord.size - 1):
            zgrid = np.concatenate((zgrid, zgrid_ + zcoord[i+1]))

    # grid for search over slowness
    if svrng[0] == svrng[1]:
        s = np.array([svrng[0]]) / 1000.
    else:
        s = np.arange(svrng[0], svrng[1] + ds, ds) / 1000.
    if not slow:
        s = 1. / (s * 1.e6)
    # extend coordinate grids and slowness grid
    sgrid = np.zeros(xgrid.size) + s[0]
    ssize = sgrid.size
    if s.size > 1:
        ygrid = np.tile(ygrid, s.size)
        xgrid = np.tile(xgrid, s.size)
        zgrid = np.tile(zgrid, s.size)
        for i in range(s.size - 1):
            sgrid = np.concatenate((sgrid, np.zeros(ssize) + s[i+1]))
    # reshape for efficient calculation
    xscoord = np.tile(scoord[:,0].reshape(n_stats, 1), (1, xgrid.size))
    yscoord = np.tile(scoord[:,1].reshape(n_stats, 1), (1, ygrid.size))
    ygrid = np.tile(ygrid, (n_stats, 1))
    xgrid = np.tile(xgrid, (n_stats, 1))
    zgrid = np.tile(zgrid, (n_stats, 1))
    sgrid = np.tile(sgrid, (n_stats, 1))
    # number of parameter combinations
    n_param = xgrid.shape[1]

    # extract number of data points
    npts = data[:, 1].size
    # construct analysis frequencies
    freq = np.arange(fmin, fmax + df, df)
    # construct time vector for sliding window 
    w_time = np.arange(0, w_length, 1./Fs)
    npts_win = w_time.size
    npts_delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    nshots = int(np.floor((npts - w_time.size) / npts_delay)) + 1

    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data = np.zeros((freq.size, n_stats, nshots), dtype=np.complex)

    # construct matrix for DFT calculation
    # dim: [number w_time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(w_time[:, None], freq[:, None].T))

    # initialize array for beamformer 
    beamformer = np.zeros(n_param)

    # calculate DFTs 
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        n = 0
        while (n * npts_delay + npts_win) <= npts:
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[n*npts_delay: n*npts_delay+npts_win][:, None],
                            np.ones((1, freq.size)))
            # mean averages over time axis
            data_freq = np.mean(np.multiply(adjust, matrice_int), axis=0)
            # fill data steering vector: ii'th station, n'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data[:, ii, n] = (data_freq / abs(data_freq)).conj().T
            n += 1


    # loop over frequencies and perform beamforming
    for ll in range(freq.size):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        K = calculate_CSDM(vect_data[ll,:,:], neig, norm)

        # calculate replica vector
        replica = np.exp(-1j * np.sqrt((xscoord - xgrid)**2 \
            + (yscoord - ygrid)**2 + zgrid**2) * 2. * np.pi * freq[ll] * sgrid)
        replica /= np.linalg.norm(replica, axis=0)
        replica = np.reshape(replica, (n_stats, n_param))

        # do phase matching
        beamformer += phase_matching(replica, K, processor)

    # normalize beamformer and reshape
    beamformer /= freq.size
    beamformer = np.reshape(beamformer, (ycoord.size, xcoord.size,
        zcoord.size, s.size), order="F")
    return ycoord, xcoord, zcoord, s*1000., beamformer