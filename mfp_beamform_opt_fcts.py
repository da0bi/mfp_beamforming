#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2022-08-23:
Created by Thoralf Dietrich <thoralf.dietrich@uni-potsdam.de> with functions from
https://github.com/fablindner/glseis/blob/master/array_analysis.py

2023-12-07:
Edited by Gerolf Vent <gvent@uni-potsdam.de>

2026-03-11:
Rewritten and optimized for speed through e.g. 
- vectorization of loops 
- reduction of sampling frequencies
by
daniel binder <daniel.binder@uni-potsdam.de>
- following fcts optimized for speed:
    - matchedfield_beamformer (some parameters removed and some added)
    - calculate_CSDM (fct not used anymore in matchedfield_beamformer)
    - annul_dominant_interferers (fct not used anymore in matchedfield_beamformer)
    - phase_matching (fct not used anymore in matchedfield_beamformer)

- new fcts included:
    - lin_log_freqs
    - nlinear_freqs
    - annul_dominant_interferers_all
    - phase_matching_fast
    - phase_matching_fast_all
    
- deleted fcts:
    - plwave_beamformer

Vectorization of code was done with AI (chatgpt).
"""

import numpy as np
import warnings


def lin_log_freqs(fmin, fmax, df):
    """
    Calculation of beamforming sampling frequencies in the range [fmin, fmax] and an 
    increasing frequency step width. Lower frequencies are sampled with a uniform df, 
    while for the higher frequencies df is continuously increasing.
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated.
    :type df: float
    :param df: optimum frequency step calculated by array phase criterion.
    
    .return: numpy.ndarray 
        freq: reduced sampling frequencies for beamforming
    """
    # compute fraction of frequency range with uniform df, and df growth parameter
    r = fmax / fmin
    linear_fraction = min(0.5, round(2 / r, 1))
    falpha = min(0.15, 0.02 * r)

    # uniform df
    f_linear_end = fmin + linear_fraction * (fmax - fmin)
    f_linear = np.arange(fmin, f_linear_end + df, df)

    # non-uniform (logarithmic) df
    f0 = f_linear[-1]

    n_log = int(np.log(fmax / f0) / np.log(1 + falpha))

    f_log = f0 * (1 + falpha) ** np.arange(1, n_log + 1)

    f_log = f_log[f_log <= fmax]

    freq = np.concatenate((f_linear, f_log))
    freq = np. round(freq, 1)
    freq = np.unique(freq)  # remove duplicates

    return freq


def nlinear_freqs(fmin, fmax, df):
    """
    Calculation of beamforming sampling frequencies in the range [fmin, fmax] and an 
    increasing frequency step width. Lower frequencies are sampled with a constant [df], 
    while for the higher frequencies [df] is continuously increasing.
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated.
    :type df: float
    :param df: optimum frequency step calculated for cmin and array aperture.
    
    .return: numpy.ndarray 
        freq: reduced sampling frequencies for beamforming
    """
    # compute linear sampling frequencies fraction and df growth parameter
    r = fmax / fmin
    linear_fraction = min(0.5, round(2 / r, 1))
    falpha = min(0.15, 0.02 * r)
    
    # uniform df 
    f_linear_end = fmin + linear_fraction * (fmax - fmin)
    f_linear = np.arange(fmin, f_linear_end + df, df)
    df = df * (1 + falpha)
    
    # non-uniform (growing) df
    f_nl = [(f_linear[-1] + df)]
    while f_nl[-1] < fmax:
        df = df * (1 + falpha)
        f = f_nl[-1] + df
        
        if f > fmax:
            break
        f_nl.append(f)
    f_nl = np.array(f_nl)
    freq = np.concatenate((f_linear, f_nl))
    freq = np.round(freq, 1)
    freq = np.unique(freq)

    return freq


def calculate_CSDM(dft_array, neig, norm):
    """
    Calculation of CSDM matrix for beamforming.
    :param dft_array: 2-Dim array containing DFTs of all stations
        and for multiple time windows. dim: [n_stations, n_windows].
    :param neig: Number of eigenvalues to project out.
    :param norm: If True, normalize CSDM matrix.

    :return: numpy.ndarray (dim: [n_stats, n_stats])
        csdm: the cross-spectral-density matrix
    """
    # CSDM matrix
    #CSDM = np.dot(dft_array, dft_array.conj().T)
    #CSDM = dft_array @ dft_array.conj().T
    CSDM = np.einsum("iw,jw->ij", dft_array, dft_array.conj())

    #if np.linalg.matrix_rank(CSDM) < dft_array.shape[0]:
    #    warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

    # annul dominant source
    if neig > 0:
        CSDM = annul_dominant_interferers(CSDM, neig, dft_array)

    # normalize
    if norm:
        #CSDM /= np.linalg.norm(CSDM)
        CSDM /= np.sqrt(np.sum(np.abs(CSDM)**2))

    return CSDM


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

    :return: numpy.ndarray (dim: [n_stats, n_stats])
        csdm: the new cross-spectral-density matrix calculated from the data after
        the projector was applied to eliminate the strongest source.
    """
    if neig <= 0:
        return CSDM
    
    # Hermitian eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(CSDM)

    # largest eigenvectors
    u = eigvecs[:, -neig:]

    # project data without forming projector
    data_proj = data - u @ (u.conj().T @ data)

    # recompute CSDM
    csdm = data_proj @ data_proj.conj().T

    return csdm


def annul_dominant_interferers_all(CSDM_all, neig):
    """
    Removes dominant eigenvectors from a stack of CSDM matrices.
    :type CSDM_all: numpy.ndarray (dim: [n_freq, n_stats, n_stats])
    :param CSDM_all: Cross-spectral density matrix stack.
    :type neig: int (>=0)
    :param neig: Number of dominant eigenvectors (noise sources) to remove

    :return: numpy.ndarray (dim: [n_freq, n_stats, n_stats])
        Projected CSDM_all matrix satck.
    """
    if neig <= 0:
        return CSDM_all

    # SVD for all frequencies
    u, s, vh = np.linalg.svd(CSDM_all)

    # dominant eigenvectors
    u_m = u[:, :, :neig]      # (n_freq, n_stats, neig)
    v_m = vh[:, :neig, :]     # (n_freq, neig, n_stats)

    # identity matrix
    n_freq, n_stats, _ = CSDM_all.shape
    I = np.eye(n_stats)[None, :, :]

    # projector
    proj = I - np.matmul(u_m, v_m)

    # apply projection
    CSDM_all_proj = np.matmul(proj, np.matmul(CSDM_all, proj.conj().transpose(0,2,1)))

    return CSDM_all_proj


def phase_matching(replica, CSDM, processor):
    """
    Does phase matching of the replica vector with the CSDM matrix.
    :type replica: numpy.ndarray (dim: [n_stats, n_param])
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations.
    :type CSDM: numpy.ndarray (dim: [n_stats, n_stats])
    :param CSDM: 2-D array CSDM matrix.
    :type processor: string ('bartlett' or 'adaptive')
    :param processor: Processor used for phase matching.

    :return: numpy.ndarray (dim: [n_param])
        the beam.
    """

    if processor == "adaptive":
        CSDM = np.linalg.inv(CSDM)

    # quadratic form for all parameters
    beam = np.einsum(
        "ip,ij,jp->p",
        replica.conj(),
        CSDM,
        replica,
        optimize=True
    )

    if processor == "bartlett":
        beam = np.abs(beam)

    elif processor == "adaptive":
        beam = np.abs(1.0 / beam)

    return beam


def phase_matching_fast(replica, CSDM, processor):
    """
    Does phase matching of the replica vector with the CSDM matrix in a faster way.
    tmp is calculated by highly optimized BLAS matrix multiplication subroutine. 
    :type replica: numpy.ndarray (dim: [n_stats, n_param])
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations.
    :type CSDM: numpy.ndarray (dim: [n_stats, n_stats])
    :param CSDM: 2-D array CSDM matrix.
    :type processor: string ('bartlett' or 'adaptive')
    :param processor: Processor used for phase matching.

    :return: numpy.ndarray (dim: [n_param])
        the beam.
    """

    if processor == "adaptive":
        CSDM = np.linalg.inv(CSDM)

    # apply CSDM
    tmp = CSDM @ replica

    # quadratic form
    beam = np.sum(replica.conj() * tmp, axis=0)

    if processor == "bartlett":
        beam = np.abs(beam)

    elif processor == "adaptive":
        beam = np.abs(1.0 / beam)

    return beam


def phase_matching_fast_all(replica_all, CSDM_all, processor):
    """
    Does phase matching of the replica vectors with the CSDM matrix in a faster way,
    and for all frequencies - avoids frequency loop. tmp calculated by highly optimized 
    BLAS matrix multiplication subroutine.
    :type replica_all: numpy.ndarray (dim: [n_freq, n_stats, n_param])
    :param replica_all: 2-D array containing the replica vectors of all parameter
        combinations and all frequencies.
    :type CSDM_all: numpy.ndarray (dim: [n_freq, n_stats, n_stats])
    :param CSDM_all: 2-D array CSDM matrix.
    :type processor: string ('bartlett' or 'adaptive')
    :param processor: Processor used for phase matching.

    :return: numpy.ndarray (dim: [n_freq, n_param])
        the beams for all frequencies.
    """

    tmp = np.einsum(
        "fsq,fqp->fsp",
        CSDM_all,
        replica_all,
        optimize=True
    )

    beam_all = np.einsum(
        "fsp,fsp->fp",
        replica_all.conj(),
        tmp,
        optimize=True
    )

    if processor == "adaptive":
        beam_all = 1.0 / beam_all

    if processor == "bartlett":
        beam_all = np.abs(beam_all)

    return beam_all

    
def matchedfield_beamformer(data, scoord, xrng, yrng, zrng, dx, dy, dz, svrng, ds,
            slow, fmin, fmax, df, freq_linear, freq_decimation, Fs, w_length, w_delay, 
            processor, neig, norm, vectorize_freq, preallocate_replica):
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
    :type df: float or None
    :param df: frequency step between fmin and fmax - if df is None the optimum df is
        calculated from the maximum slowness
    :type freq_linear: boolean
    :param freq_linear: if True, frequencies range with constant frequency step
                        if False, increasing frequency steps for higher frequencies
                        (-> see nlinear_freq and lin_log_freq fcts)
    :type freq_decimation: int (>0) or None
    :param freq_decimation: frequency decimation factor applied to the frequency axis when evaluating
        the beamformer. Only every `freq_decimation`-th frequency is used while the skipped frequencies
        are accounted for by bandwidth weighting. if not None 'freq_linear' parameter is set to True.
        Better alternative to a growing df for higher frequencies. A too large df can introduce phase
        decorrelation - incoherent beam patterns for neighbouring frequencies, due to phase changes > pi/2.
        how too choose: array aperture      freq_decimation factor
                        small (< 1km)               2-3
                        medium (~3km)               3-6
                        very large (>10km)          6-10
        beam patterns vary slowly with frequency. theoretically the decimation factor is limited so that 
        the effective frequency step remains smaller than the maximum step allowed by the array phase stability 
        criterion: df_max = 1 / (4 * array_aperture * max_slowness), 
        while reducing computational costs. this ensures that steering phases remain coherent across the 
        array and prevents beamforming artefacts due to excessive frequency spacing.
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type processor: string ("bartlett" or "adaptive")
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vector. see Corciulo et al., 2012
    :type neig: integer (>=0)
    :param neig: number of dominant CSDM eigenvectors to annul from the data - suppresses
        strong sources.
    :type norm: boolean
    :param norm: if True, beam power is normalized.
    :type vectorize_freq: boolean
    :param vectorize_freq:  if True, last remaining loop over the sample frequencies is also
                            vectorized. all replicas and beams are computed at once - more
                            memory intense.
                            if False, replica and beams are computed inside the frequency loop
                            - memory-friendly.
    :type preallocate_replica: boolean
    :param preallocate_replica: if True, replica is preallocated and allocated memory is reused
                                in the frequency loop - memory-friendly.
                                if False, each replica is allocated at an individual memory location
                                during the single frequency loops.

    :return: four numpy arrays:
        xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
        ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
        zcoord: grid coordinates in z-direction (dim: [number z-grid points, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number y-grid points, number x-grid points, number z-grid points, number cs])
    """

    # number of stations
    n_stats = data.shape[1]
    
    # -------------------------------------------------
    # spatial grid
    # -------------------------------------------------

    xcoord = np.array([xrng[0]]) if xrng[0] == xrng[1] else np.arange(xrng[0], xrng[1] + dx, dx)
    ycoord = np.array([yrng[0]]) if yrng[0] == yrng[1] else np.arange(yrng[0], yrng[1] + dy, dy)
    zcoord = np.array([zrng[0]]) if zrng[0] == zrng[1] else np.arange(zrng[0], zrng[1] + dz, dz)

    s = np.array([svrng[0]]) if svrng[0] == svrng[1] else np.arange(svrng[0], svrng[1] + ds, ds)
    s = s / 1000.
    if not slow:
        s = 1. / (s * 1.e6)

    # parameter grid
    X, Y, Z, S = np.meshgrid(xcoord, ycoord, zcoord, s, indexing="ij")

    xg = X.ravel(order="F")
    yg = Y.ravel(order="F")
    zg = Z.ravel(order="F")
    sg = S.ravel(order="F")

    n_param = xg.size

    # -------------------------------------------------
    # station coordinates
    # -------------------------------------------------

    xs = scoord[:, 0][:, None]
    ys = scoord[:, 1][:, None]

    xs2 = xs**2
    ys2 = ys**2

    grid_norm2 = xg**2 + yg**2 + zg**2

    # dot product between station coords and grid
    dot = xs * xg + ys * yg

    # squared distance, and distance
    dist2 = xs2 + ys2 + grid_norm2 - 2 * dot
    dist = np.sqrt(np.maximum(dist2, 0)) # avoids slightly negative distances due to floating point rounding
    
    # -------------------------------------------------
    # compute frequencies to process
    # -------------------------------------------------
        
    # if no df defined, calculate through array phase stability criterion:
    # cmin / (4 * L_array)
    # cmin....minimum phase velocity [m/s] -> (1/max_slowness)*1000
    # L_array...array aperture [m]
    
    if df is None:

        # calculate optimum frequency step over cmin and array aperture estimation (L_array)
        dx = np.max(scoord[:,0]) - np.min(scoord[:,0])
        dy = np.max(scoord[:,1]) - np.min(scoord[:,1])

        L_array = max(dx, dy)
    
        # compute minimum phase velocity 
        if slow:
            cmin = (1 / np.max(svrng)) * 1000
            cmin = (cmin // 500) * 500

        elif not slow:
            cmin = np.min(svrng)
            cmin = (cmin // 500) * 500

        df = cmin / (4 * L_array)
        df = round(df * 10) / 10

    else:
        
        df= round(df * 10) / 10
        
    # compute bandwidth and number of sampling frequencies based on a constant df
    bw = fmax - fmin
    nf = int(round(bw / df)) + 1

    # activate freq_linear if freq_decimation is applied
    if freq_decimation is not None:
        
        freq_linear = True
    
    # compute sampling frequencies 
    if freq_linear or nf <= 201:

        freq = np.arange(fmin, fmax + df, df)
        
        if freq_decimation is not None and nf > 201:

            freq = freq[::freq_decimation]
            
            df = np.gradient(freq)
            
    else:

        #freq = nlinear_freqs(fmin, fmax, df)
        freq = lin_log_freqs(fmin, fmax, df)
       
        # calculate actual df for final beamformer normalization
        df = np.gradient(freq)
          
    # -------------------------------------------------
    # sliding window preparation
    # -------------------------------------------------

    npts = data.shape[0]

    w_time = np.arange(0, w_length, 1./Fs)

    npts_win = w_time.size
    npts_delay = int(w_delay * Fs)

    nshots = int(np.floor((npts - npts_win) / npts_delay)) + 1

    starts = np.arange(0, nshots * npts_delay, npts_delay)
    idx = starts[:, None] + np.arange(npts_win)

    data_win = data[idx, :]
    data_win = np.transpose(data_win, (1, 2, 0))
    
    # -------------------------------------------------
    # precompute DFT matrices for all frequencies
    # -------------------------------------------------

    matrice_int = np.exp(2j * np.pi * np.outer(w_time, freq))

    vect_data = np.einsum(
        "tf,tsn->fsn",
        matrice_int,
        data_win
    ) / npts_win

    #vect_data = np.conj(vect_data / np.abs(vect_data))
    
    # if vect_data == 0, division produces NaN
    # normalize DFT phase safely
    mag = np.abs(vect_data)
    mag[mag == 0] = 1
    vect_data = np.conj(vect_data / mag)
    
    # --------------------------------------------------------------
    # precompute cross spectral density matrices for all frequencies
    # --------------------------------------------------------------
    '''
    CSDM_all = np.einsum(
        "fsn,fjn->fsj",
        vect_data,
        vect_data.conj(),
        optimize=True
    )
    '''
    # use fast BLAS matrix multiplication for CSDM
    CSDM_all = vect_data @ vect_data.conj().transpose(0,2,1)

    if neig > 0:
        CSDM_all = annul_dominant_interferers_all(CSDM_all, neig)

    if norm:
        CSDM_all /= np.linalg.norm(CSDM_all, axis=(1, 2), keepdims=True)

    if processor == "adaptive":
        CSDM_all = np.linalg.inv(CSDM_all)
    
    # -------------------------------------------------
    # phase matching & beamforming
    # -------------------------------------------------

    # precompute omega
    omega = 2 * np.pi * freq

    # precompute geometric delay term
    # element-wise broadcasting (distances x slownesses)
    dist_sg = dist * sg

    if vectorize_freq:
        # ------------------------------------------------------------
        # last remaining loop over all frequencies is also vectorized.
        # replicas and beams are computed for all frequencies at once
        # to then build the beamformer from all the calculated beams.
        # ------------------------------------------------------------

        replica_all = np.exp(-1j * omega[:, None, None] * dist_sg[None, :, :])

        # normalize steering vectors - not necessary, steering vectors already have identical norms
        replica_all /= np.linalg.norm(replica_all, axis=1, keepdims=True)

        beam_all = phase_matching_fast_all(replica_all, CSDM_all, processor)
        
        # normalize (average) all calculated beams to build beamformer         
        if np.isscalar(df):
            # uniform df 
            beamformer = np.mean(beam_all, axis=0)
        
        else:
            # non-uniform df - bandwidth-weighted averaging is necessary  
            beamformer = np.sum(beam_all * df[:,None], axis=0) 
            
            beamformer /= np.sum(df)
        
    else:
        # -----------------------------------------------------------------
        # loop over frequencies and compute replicas and beams individually
        # -----------------------------------------------------------------

        # initialize array for all beams
        beam_all = np.zeros(n_param, dtype=float)

        if preallocate_replica:
            # preallocate replica before the frequency loop
            # replica must have the same shape as dist_sg
            replica = np.empty_like(dist_sg, dtype=np.complex128)

        # loop over frequencies and perform beamforming
        for ll in range(freq.size):
            # compute replica vectors from phases
            if preallocate_replica:
                # reusing the memory of the preallocated replica
                # -> reduces memory overhead, but can be slower
                np.exp(-1j * omega[ll] * dist_sg, out=replica)
            else:
                # allocating replica for every loop run
                # -> can be faster if there is no memory issue
                replica = np.exp(
                    -1j 
                    * omega[ll] 
                    * dist_sg
                    )
            
            # normalize and reshape replica matrix 
            replica /= np.linalg.norm(replica, axis=0)
            replica = np.reshape(replica, (n_stats, n_param))
            
            # calculate cross-spectral density matrix
            # dim: [number of stations X number of stations]
            CSDM = CSDM_all[ll]

            if np.isscalar(df):
                # uniform df        
                beam_all += phase_matching_fast(replica, CSDM, processor)
            
            else:
                # non-uniform df - beam is multiplied by corresponding df which is the first 
                # necessary step for bandwidth-weighted averaging to build the final beamformer 
                beam_all += phase_matching_fast(replica, CSDM, processor) * df[ll]

        # normalize all calculated beams to build beamformer                     
        if np.isscalar(df):
            # uniform df
            beamformer = beam_all / freq.size
            
        else:
            # non-uniform df - all beams are normalized by the df-sum as the second and 
            # final step of the bandwidth-weighted averaging to build the final beamformer  
            beamformer = beam_all / np.sum(df)

    # final beamformer build
    beamformer = beamformer.reshape(
    xcoord.size,
    ycoord.size,
    zcoord.size,
    s.size,
    order="F"
    )

    return xcoord, ycoord, zcoord, s*1000., beamformer