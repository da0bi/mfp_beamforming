#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shelve
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import geopandas as gpd
import imageio.v2 as imageio


def imp_shelve_file(directory, file_to_load):
    """
    Loads beam_data dictionary computed by the matchedfield_beamformer fct.
    
    beam_data structure:
    beam_data = {
        'scnl': cur_scnl,
        'start_time': start_time,
        'xcoord': xcoord,
        'ycoord': ycoord,
        'zcoord': zcoord,
        'c': c,
        'beamformer': beamformer
        }
        
    :return: beam_data dicitonary
    """
    
    filepath_to_load = os.path.join(
        directory,
        file_to_load,
    )
    bf_data = {}

    # Open shelve file
    db = shelve.open(filepath_to_load)
    # Update the local dictionary with the loaded one
    bf_data.update(db)
    # Close shelve file
    db.close()

    return bf_data


def bf_opt_arr_per_t(bf_data, mode):
    """
    Computes and saves the mean beamformer array for each beam_data dicitionary. 
    Extracts the optimum z-coordinate and slowness for each beamformer timestep.
    
    :type bf_data: dictionary
    :param bf_data: see imp_shelve_file fct for dictionary structure
    :type mode: string ('mean' or 'max')
    :param mode: defines the criterion of choosing the optimum beamformer for each timestep.
        'mean': beamfomer with highest mean value, 'max': beamfomer with highest max value    

    :return:    best_per_t: 
                    dictionary with optimum z and slowness, and highest semblance value for 
                    each timestep.
                mean_bf_array:
                    numpy array of the mean beamformer
                xymax:
                    list of x, y coordinates of the max semblance value for each timestamp.
    """
    
    best_per_t = {}
    bf_arrays = []

    t = next(iter(bf_data))
    zcoord = bf_data[t]['zcoord']
    svals = bf_data[t]['c']

    if mode == 'max':
        
        xcoord = bf_data[t]['xcoord']
        ycoord = bf_data[t]['ycoord']
        xymax = []

    print('datetime', 'opt_z', 'opt_s', 'max_mean_bf', sep="\t")

    for t in bf_data.keys():
        
        arr = bf_data[t]['beamformer']    # shape(x-range, y-range, z-range, slowness-range)

        if mode == 'mean':
            
            arr_filtered = arr.mean(axis=(0,1))    # shape(z-range, slowness-range)
            z_idx, s_idx = np.unravel_index(arr_filtered.argmax(), arr_filtered.shape)

        elif mode == 'max':
            
            arr_filtered = arr.max(axis=(0,1))    # shape(z-range, slowness-range)
            z_idx, s_idx = np.unravel_index(arr_filtered.argmax(), arr_filtered.shape)
            
            # save datetime and coordinates of max value
            xmax, ymax = max_xy_coord(arr[:,:,z_idx,s_idx], xcoord, ycoord)
            xymax.append((t, xmax, ymax))
             
        # save the beamformer array with the highest mean or max value for each time step
        bf_arrays.append(arr[:,:,z_idx,s_idx])
        
        # assign values
        best_z = zcoord[z_idx]
        best_s = svals[s_idx]
        best_arr = arr_filtered[z_idx, s_idx]

        best_per_t[t] = {
            "z": best_z,
            "s": best_s,
            "arr_opt": best_arr
        }
        
        # print values
        print(t, best_z, best_s, best_arr, sep="\t")
    
    # calculate the mean beamformer array of all the beamformer array
    # with the highest means or maxs for each time step
    mean_bf_array = np.mean(bf_arrays, axis=0)

    return best_per_t, mean_bf_array, xymax


def max_xy_coord(arr, xcoord, ycoord):
    """
    Looks up the x, y coordinates of the array'smax value.
    
    :type arr: numpy.ndarray
    :param arr:  beamformer array for a timestep
    :type xcoord, ycoord: list
    :param xcoord, ycoord: list of x,y-coordinates    

    :return:    max_x, max_y: 
                    x,y-coordinates of the maximum array value.
    """
    # find the flat index of the max value
    flat_idx = np.argmax(arr)

    # unravel the index into 2D indices (ix, iy)
    ix, iy = np.unravel_index(flat_idx, arr.shape)

    # retrieve the actual coordinate values from your arrays
    max_x = xcoord[ix]
    max_y = ycoord[iy]

    return max_x, max_y


def plot_bf_array(mean_bf_array, semb_max, semb_min, xcoord, ycoord, scoord, xymax, 
                  shp_path, title, png_name, png_dir):
    """
    Plots the mean beamformer array of each bf_data (*db) file. 
    Plots the locations of the array's maximum values for each timestep.
    Plots the glacier outline.
    
    :type mean_bf_array: numpy.ndarray
    :param mean_bf_array: the mean beamformer
    :type semb_max, semb_min: float
    :param semb_max, semb_min: maximum and minimum semblance values to plot
    :type xcoord, ycoord: list
    :param xcoord, ycoord: xy-coordinates of the array   
    :type scoord: list
    :param scoord: xy-coordinates of the seismic stations
    :type xymax: list
    :param xymax: list of x, y coordinates of the max semblance value for each timestamp.
    :type shp_path: string
    :param shp_path: path to glacier outline shapefile
    :type title: string
    :param title: plot title
    :type png_name: string
    :param png_name: name for saving png file
    :type png_dir: string
    :param png_dir: directory path for saving png file
    """
    
    # get station coordinates    
    xs = scoord[:, 0]
    ys = scoord[:, 1]

    # load apo shapefile
    gdf = gpd.read_file(shp_path)

    # initiate figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # plot beamformer array
    im = ax.imshow(
    mean_bf_array.T, # imshow expects (y, x) 
    origin="lower",
    extent=[
        xcoord.min(), xcoord.max(),
        ycoord.min(), ycoord.max()
    ],
    cmap="inferno",
    aspect="equal"
    )
    
    # plot formating
    im.set_clim(semb_min, semb_max)
    plt.colorbar(im, ax=ax, label="Semblance")

    # plot shapefile
    gdf.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=1)

    # overlay station coordinates
    ax.scatter(
        xs, 
        ys,
        marker='^',
        c="white",
        s=50,
        edgecolor="black",
        label="Stations",
        zorder=3
        )

    # overlay max semblance value of each time step
    if xymax is not None:
        t, xmax, ymax = zip(*xymax)
        ax.scatter(
            xmax, 
            ymax,
            c=range(len(t)),
            cmap='inferno',
            s=30,
            edgecolor="white",
            label="max semblance",
            zorder=3
        )

    # plot formating and labeling
    ax.set_xlim(xcoord.min()-75, xcoord.max()+75)
    ax.set_ylim(ycoord.min()-75, ycoord.max()+75)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    ax.set_title(title)

    # save the plot
    png_path = png_dir / f"{png_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")


###################
# DEFINE PARAMETERS
###################
# Define path for loading the data
directory = Path("/home/db/Projects/APO_Monitoring_2023/10_psysmon/output/mfp_beamforming")
proc_folder = "test_run10"
sfolder = "beam"
directory = directory / proc_folder / sfolder
# Define station, component, network and channel for the data to load
#station = "AP233"
#comp = "DPZ"
#network = "AP"
#loc = "C"
# Define year and day of year to process
year = "2023"
doy = "182"
year_doy = year + "_" + doy
# construct path for loading the data
#directory = directory / station / comp / year_doy
directory = directory / year_doy

# station coordinates
scoord = np.array([
    [4.88863554e+05, 8.28159521e+06],
    [4.88873332e+05, 8.28157018e+06],
    [4.88848224e+05, 8.28157333e+06],
    [4.87705465e+05, 8.28305286e+06],
    [4.87693803e+05, 8.28302934e+06],
    [4.87680495e+05, 8.28305106e+06],
    [4.85930291e+05, 8.28385403e+06],
    [4.85949843e+05, 8.28386520e+06],
    [4.85946872e+05, 8.28384478e+06]
])

# path to shapefile
shp_path = "/home/db/Projects/APOlsen/01_QGis/03_APO_shp/APO_utm.shp"

# set semblance limits for bf plotting
semb_max = 0.4 
semb_min = 0.2

##########################################
# LOAD DATA, CALCULATE AND PLOT BEAMFORMER
##########################################
for fpath in sorted(directory.iterdir()):
    if fpath.is_file() and fpath.suffix == ".db":
        print(fpath)

        # load the data
        bf_data = imp_shelve_file(fpath.parent, fpath.name)

        # calculate the mean beamformer array of all the beamformer array with the highest means or maxs per time step
        best_per_t, mean_bf_array, xymax = bf_opt_arr_per_t(bf_data, mode='max')

        # plotting
        # take random timestamp and extract the x and y coordinates for plotting
        t = list(bf_data.keys())[0]
        xcoord = bf_data[t]['xcoord']
        ycoord = bf_data[t]['ycoord']
        
        # parse string to datetime
        # adjust the format string for the title
        t_dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
        t_next = t_dt + timedelta(hours=1)
        title = f"{t_dt:%Y-%m-%d} {t_dt:%H}:00 – {t_next:%H}:00"
        png_name = f"{t_dt:%Y-%m-%d}_{t_dt:%H}_{t_next:%H}"

        # make subfolder for saving the plots
        png_dir = directory / "gif_frames"
        png_dir.mkdir(exist_ok=True)

        plot_bf_array(mean_bf_array,
                      semb_max,
                      semb_min,
                      xcoord,
                      ycoord,
                      scoord,
                      xymax,
                      shp_path,
                      title,
                      png_name,
                      png_dir
                      )

# Create a GIF animation of the beamformer array over time
gif_name = f"{t_dt:%Y-%m-%d}_animation.gif"
gif_path = png_dir / gif_name
frames = [imageio.imread(png) for png in sorted(png_dir.glob("*.png"))]
imageio.mimsave(
    gif_path,
    frames,
    duration=5,
    loop=0
)
