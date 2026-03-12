#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2022-08-23:
Created by Thoralf Dietrich <thoralf.dietrich@uni-potsdam.de> with functions from
https://github.com/fablindner/glseis/blob/master/array_analysis.py

2023-12-07:
Edited by Gerolf Vent <gvent@uni-potsdam.de>

2026-03-11:
Rewritten and optimized for speed through e.g. vectorization of loops by
daniel binder <daniel.binder@uni-potsdam.de>
- following fcts optimized for speed:
    - matchedfield_beamformer (some parameters removed and added)
    - calculate_CSDM (fct not used anymore in matchedfield_beamformer)
    - annul_dominant_interferers (fct not used anymore in matchedfield_beamformer)
    - phase_matching (fct not used anymore in matchedfield_beamformer)

- new fcts included:
    - nlinear_freqs
    - annul_dominant_interferers_all
    - phase_matching_fast
    - phase_matching_fast_all

Vectorization of code was done with AI (chatgpt).
"""

import numpy as np
import warnings


def nlinear_freqs(fmin, fmax, df, linear_fraction, falpha):
    """
    Calculate frequency range with increasing frequency steps for higher frequencies.
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type df: float
    :param df: optimum frequency step calculated for cmin and array aperture
    :type linear_fraction: float
    :param linear_fraction: starting fraction with constant frequency steps df (0-1)
    :type falpha: float
    :param falpha: frequency step df growth factor (0-1)
    """
    if linear_fraction is None or falpha is None:

        freq = np.arange(fmin, fmax + df, df)

    else:
        f_linear_end = fmin + linear_fraction * (fmax - fmin)
        f_linear_end = round(f_linear_end * 20) / 20
        f_linear = np.arange(fmin, f_linear_end + df, df)

        df = df * (1 + falpha)

        f_nl = [round((f_linear[-1] + df) * 20) / 20]

        while f_nl[-1] < fmax:

            df = df * (1 + falpha)

            f = f_nl[-1] + df

            f_nl.append(round(f * 20) / 20)

        f_nl = np.array(f_nl)

        freq = np.concatenate((f_linear, f_nl))

    return freq


def calculate_CSDM(dft_array, neig, norm):
    """
    Calculate CSDM matrix for beamforming.
    :param dft_array: 2-Dim array containing DFTs of all stations
        and for multiple time windows. dim: [n_stations, n_windows]
    :param neig: Number of eigenvalues to project out.
    :param norm: If True, normalize CSDM matrix.
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

    :return: numpy.ndarray
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
    Remove dominant eigenvectors from a stack of CSDM matrices.
    :type CSDM_all: numpy.ndarray
    :param CSDM_all: Cross-spectral density matrix stack (dim: [n_freq, n_stats, n_stats])
    :type neig: int
    :param neig: Number of dominant eigenvectors (noise sources) to remove

    :return: numpy.ndarray
        Projected CSDM_all matrix satck (dim: [n_freq, n_stats, n_stats])
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
    Do phase matching of the replica vector with the CSDM matrix.
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations (dim: [n_stats, n_param])
    :param CSDM: 2-D array CSDM matrix (dim: [n_stats, n_stats])
    :param processor: Processor used for phase matching. bartlett or adaptive.

    :return: numpy.ndarray
        beam
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
    Do phase matching of the replica vector with the CSDM matrix in a faster way.
    (tmp calculated by extremly optimized BLAS matrix multiplication)
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations (dim: [n_stats, n_param])
    :param CSDM: 2-D array CSDM matrix (dim: [n_stats, n_stats])
    :param processor: Processor used for phase matching. bartlett or adaptive.
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
    Do phase matching of the replica vector with the CSDM matrix in a faster way.
    (tmp calculated by extremly optimized BLAS matrix multiplication)
    Applies fast phase matching to all frequencies - no frequency loop necessary.
    :param replica_all: 2-D array containing the replica vectors of all parameter
        combinations and all frequencies(dim: [n_freq, n_stats, n_param])
    :param CSDM_all: 2-D array CSDM matrix (dim: [n_freq, n_stats, n_stats])
    :param processor: Processor used for phase matching. bartlett or adaptive.
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
            slow, fmin, fmax, cmin, freq_linear, Fs, w_length, w_delay, processor,
            neig, norm, precompute_replica, preallocate_replica):
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
    :type cmin: float
    :param cmin: minimum apparent phase velocity (m/s)
    :type freq_linear: boolean
    :param freq_linear: if True, frequencies range with constant frequency step
                        if False, increasing frequency steps for higher frequencies
                        (-> see nlinear_freq fct)
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type processor: string
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vector. see Corciulo et al., 2012
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
        enables to suppress strong sources.
    :type norm: boolean
    :param norm: if True, beam power is normalized
    :type precompute_replica: boolean
    :param precompute_replica:  if True, replica is precomputed (memory intense for bigger arrays)
                                if False, replica is computed for every frequency loop
    :type preallocate_replica: boolean
    :param preallocate_replica: if True, replica is preallocated and memory reused in the frequency loop
                                if False, replica is allocated and stored at another place for every
                                frequency loop

    :return: four numpy arrays:
        xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
        ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number y-grid points, number x-grid points, number cs])

    Parameters differing from plwave_beamformer fct:
    xrng, yrng, zrng, dx, dy, dz, svrng, ds
    """

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
    X, Y, Z, S = np.meshgrid(xcoord, ycoord, zcoord, s, indexing="xy")

    xg = X.ravel()
    yg = Y.ravel()
    zg = Z.ravel()
    sg = S.ravel()

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

    # calculate optimum frequency step over cmin and array aperture estimation (L_array)
    dx = np.max(scoord[:,0]) - np.min(scoord[:,0])
    dy = np.max(scoord[:,1]) - np.min(scoord[:,1])

    L_array = max(dx, dy)

    if cmin is None:
         raise ValueError("Cmin (m/s) must be provided.")

    df = cmin / (4 * L_array)
    df = np.round(df * 20) / 20

    if freq_linear:

        freq = np.arange(fmin, fmax + df, df)

    else:

        freq = nlinear_freqs(fmin, fmax, df,
                        linear_fraction=0.4,
                        falpha=0.05
                        )

    n_freq = freq.size

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
    # use fast BLAS matrix multiply for CSDM
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

    # precompute geometric delay term (distances x slownesses)
    dist_sg = dist * sg

    if precompute_replica:
        # -------------------------------------------------------
        # precompute replica vectors beamform for all frequencies
        # -------------------------------------------------------

        replica_all = np.exp(-1j * omega[:, None, None] * dist_sg[None, :, :])

        # normalize steering vectors - not necessary, steering vectors already have identical norms
        #replica_all /= np.linalg.norm(replica_all, axis=1, keepdims=True)

        beam_all = phase_matching_fast_all(replica_all, CSDM_all, processor)

        beamformer = np.mean(beam_all, axis=0)

    else:
        # ----------------------------------------------------------------------------------
        # loop over frequencies - memory efficient
        # avoids calculation and allocation of replica_all
        # ----------------------------------------------------------------------------------

        # initiate beamformer
        beamformer = np.zeros(n_param, dtype=float)

        if preallocate_replica:
            # preallocate replica before the frequency loop
            # replica must have the same shape as dist_sg
            replica = np.empty_like(dist_sg, dtype=np.complex128)

        for ll in range(freq.size):
            # compute replica vectors from phases
            if preallocate_replica:
                # reusing the memory of the preallocated replica
                # reduces memory overhead, but can be slower
                np.exp(-1j * omega[ll] * dist_sg, out=replica)
            else:
                # allocating replica for every loop run
                # can be faster if there is no memory issue
                replica = np.exp(-1j * omega[ll] * dist_sg)

            # normalize steering vectors - not necessary, steering vectors already have identical norms
            #replica /= np.linalg.norm(replica, axis=0, keepdims=True)

            CSDM = CSDM_all[ll]

            beam = phase_matching_fast(replica, CSDM, processor )

            beamformer += np.abs(beam)

        beamformer /= freq.size

    beamformer = beamformer.reshape(
    ycoord.size,
    xcoord.size,
    zcoord.size,
    s.size,
    order="F"
    )

    return ycoord, xcoord, zcoord, s*1000., beamformer
