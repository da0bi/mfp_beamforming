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

##
def imp_shelve_file(directory, file_to_load):
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

def bf_opt_arr_per_t(bf_data):
    # Extract the optimum z-coordinate and slowness to receive 
    # highest mean beamformer value for each time step
    # save beamformer array with highest mean value for each time step  
    best_per_t = {}
    bf_arrays = []    

    print('datetime', 'opt_z', 'opt_s', 'max_mean_bf', sep="\t")   

    for t in bf_data.keys():
        arr = bf_data[t]['beamformer']    # shape(x-range, y-range, z-range, slowness-range)
        zcoord = bf_data[t]['zcoord']
        s_vals = bf_data[t]['c']

        arr_mean = arr.mean(axis=(0,1))    # shape(z-range, slowness-range)

        z_idx, s_idx = np.unravel_index(arr_mean.argmax(), arr_mean.shape)

        # save the beamformer array with the highest mean value for each time step
        bf_arrays.append(arr[:,:,z_idx,s_idx])

        best_z = zcoord[z_idx]
        best_s = s_vals[s_idx]
        best_arr = arr_mean[z_idx, s_idx]

        best_per_t[t] = {
            "z": best_z,
            "s": best_s,
            "arr_mean": best_arr
        }

        print(t, best_z, best_s, best_arr, sep="\t")
    # calculate the mean beamformer array of all the beamformer array 
    # with the highest means for each time step
    mean_bf_array = np.mean(bf_arrays, axis=0)
    
    return best_per_t, mean_bf_array

def plot_bf_array(mean_bf_array, semb_max, semb_min, xcoord, ycoord, scoord, shapefile_path, title, png_name, png_dir):
    # plot the mean beamformer array of all the beamformer array with the highest means for each time step
    xs = scoord[:, 1]
    ys = scoord[:, 0]

    # load apo shapefile
    gdf = gpd.read_file(shp_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    pcm = ax.pcolormesh(ycoord, xcoord, mean_bf_array, shading="auto")
    pcm.set_clim(semb_min, semb_max)
    plt.colorbar(pcm, ax=ax, label="Semblance")

    # plot shapefile
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    # overlay station coordinates
    ax.scatter(ys, xs, c="black", s=30, edgecolor="black", label="Stations", zorder=3)

    ax.set_xlim(ycoord.min(), ycoord.max())
    ax.set_ylim(xcoord.min(), xcoord.max()) 

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    ax.set_title(title)
    
    # save the plot
    png_path = png_dir / f"{png_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    
    #plt.show()


# Define path for loading the data
directory = Path("/home/db/Projects/APO_Monitoring_2023/10_psysmon/output/mfp_beamforming_test")
proc_folder = "smi-up.db-psysmon-apo23-mfp_beamforming_20260209_141822_044687-time_window_looper"
sfolder = "beam"
directory = directory / proc_folder / sfolder
# Define station, component, network and channel for the data to load
station = "AP233"
comp = "DPZ"
network = "AP"
loc = "C"
# Define year and day of year to process
year_doy = "2023_182"
# construct path for loading the data
directory = directory / station / comp / year_doy

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

for fpath in sorted(directory.iterdir()):
    if fpath.is_file() and fpath.suffix == ".db":
        print(fpath)
        
        # load the data
        bf_data = imp_shelve_file(fpath.parent, fpath.name)

        # calculate the mean beamformer array of all the beamformer array with the highest means for each time step
        best_per_t, mean_bf_array = bf_opt_arr_per_t(bf_data)    

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

