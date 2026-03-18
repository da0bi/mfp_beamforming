#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shelve
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import geopandas as gpd

import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.windows import from_bounds

from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator

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
        
    Returns: 
    --------
    bf_data : dictionary
        Beamformer dictionary. 
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
    
    Parameters:
    -----------
    bf_data : dictionary
        See imp_shelve_file fct for dictionary structure.
    mode : string ('mean' or 'max')
        Defines the criterion of choosing the optimum beamformer for each timestep.
        'mean': beamfomer with highest mean value, 'max': beamfomer with highest max value    

    Returns:
    --------
    best_per_t : list 
        Dictionary with optimum z and slowness, and highest semblance value for each timestep.
    mean_bf_array : numpy.ndarray
        Array of the mean beamformer.
    xymax : list
        Datetime and x, y coordinates of the max semblance value for each timestamp.
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
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Beamformer array for one timestep.
    coord, ycoord : list
        x,y-coordinates of the array.    

    Returns:
    --------
    max_x, max_y : list
        x,y-coordinates of the maximum array values.
    """
    # find the flat index of the max value
    flat_idx = np.argmax(arr)

    # unravel the index into 2D indices (ix, iy)
    ix, iy = np.unravel_index(flat_idx, arr.shape)

    # retrieve the actual coordinate values from your arrays
    max_x = xcoord[ix]
    max_y = ycoord[iy]

    return max_x, max_y


def plot_bf_array(
    mean_bf_array, semb_max, semb_min,
    xcoord, ycoord, scoord,
    title, png_name, png_dir,
    shp_path, ortho_path, xymax, 
    background, bf_crs
):
    """
    Plot the mean beamformer array on top of optional backgrounds.
    Locations of the maximum semblance values can be also plotted.

    Parameters
    ----------
    mean_bf_array : np.ndarray
        2D beamformer array. (dim: [ny, nx])
    semb_max, semb_min : float
        Max and min semblance values for plotting.
    xcoord, ycoord : numpy.ndarray
        Coordinates of beamformer grid. (dim: [nx], [ny])
    scoord : numpy.ndarray
        Station coordinates. (dim: [n_stations, 2])
    title : str
        Plot title.
    png_name : str
        Output PNG file name.
    png_dir : str or Path
        Directory to save PNG.
    shp_path : str or None
        Path to shapefile. (optional)
    ortho_path : str or None
        Path to ortho image. (optional)
    xymax : list or None
        List of tuples [(t, x_max, y_max), ...] of beamformer maxima. (optional)
    background : str
        Which background to show: "shp", "ortho", "both", "none".
    bf_crs : str
        CRS of the beamformer coordinates (e.g. "EPSG:32627").
    """

    xs = scoord[:, 0]
    ys = scoord[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # -----------------------------
    # Plot background
    # -----------------------------
    if background in ("shp", "both") and shp_path is not None:
        
        gdf = gpd.read_file(shp_path)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    if background in ("ortho", "both") and ortho_path is not None:
        
        with rasterio.open(ortho_path) as src:
            ortho = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            bounds = src.bounds
            raster_crs = src.crs
        
        # CRS check
        bf_crs_obj = CRS.from_string(bf_crs)
        if raster_crs != bf_crs_obj:
            raise ValueError(
                f"CRS mismatch:\nBeamformer CRS: {bf_crs_obj}\nRaster CRS: {raster_crs}"
            )
        
        ax.imshow(
            ortho,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            origin="upper",
        )

    # ------------------------------
    # Optional masking of beamformer
    # ------------------------------
    if background in ("ortho", "both") and ortho_path is not None:
        
        semb_min = 0.35
        masked_bf = np.ma.masked_where(mean_bf_array < semb_min, mean_bf_array)
        
    else:
        
        masked_bf = mean_bf_array

    # -------------------------------
    # Plot beamformer (original grid)
    # -------------------------------
    im = ax.imshow(
        masked_bf.T,  # transpose so (y, x)
        origin="lower",
        extent=[xcoord.min(), xcoord.max(), ycoord.min(), ycoord.max()],
        cmap="inferno",
        alpha=0.5 if background in ("ortho", "both") and ortho_path is not None else 1.0,
        aspect="equal",
    )
    im.set_clim(semb_min, semb_max)
    plt.colorbar(im, ax=ax, label="Semblance")

    # -----------------------------
    # Overlay stations
    # -----------------------------
    ax.scatter(
        xs,
        ys,
        marker="^",
        c="white",
        s=50,
        edgecolor="black",
        label="Stations",
        zorder=3,
    )

    # -----------------------------
    # Overlay max beamformer locations
    # -----------------------------
    if xymax is not None:
        t, xmax, ymax = zip(*xymax)
        ax.scatter(
            xmax,
            ymax,
            c=range(len(t)),
            cmap="inferno",
            s=30,
            edgecolor="white",
            label="Max semblance",
            zorder=3,
        )

    # -----------------------------
    # Axes formatting
    # -----------------------------
    ax.set_xlim(xcoord.min() - 75, xcoord.max() + 75)
    ax.set_ylim(ycoord.min() - 75, ycoord.max() + 75)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)

    # -----------------------------
    # Save figure
    # -----------------------------
    png_dir = Path(png_dir)
    png_dir.mkdir(exist_ok=True, parents=True)
    png_path = png_dir / f"{png_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

# path to orthophoto
ortho_path = "/home/db/Projects/APOlsen/01_QGis/2016_SDFE_DEM/APO_dem/merged_utm_cropped.tif"

# set semblance limits for bf plotting
semb_max = 0.4 
semb_min = 0.2

# take random timestamp and extract the x and y coordinates
t = list(imp_shelve_file(directory, sorted(directory.glob("*.db"))[0].name).keys())[0]
xcoord = imp_shelve_file(directory, sorted(directory.glob("*.db"))[0].name)[t]['xcoord']
ycoord = imp_shelve_file(directory, sorted(directory.glob("*.db"))[0].name)[t]['ycoord']


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

        # parse string to datetime
        t = list(bf_data.keys())[0]
        t_dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
        t_next = t_dt + timedelta(hours=1)
        
        # adjust the format string for the plot title and png file name 
        title = f"{t_dt:%Y-%m-%d} {t_dt:%H}:00 – {t_next:%H}:00"
        png_name = f"{t_dt:%Y-%m-%d}_{t_dt:%H}_{t_next:%H}"

        # make subfolder for saving the plots
        png_dir = directory / "gif_frames"
        png_dir.mkdir(exist_ok=True)

        plot_bf_array(mean_bf_array, semb_max, semb_min,
                      xcoord, ycoord, scoord,
                      title, png_name, png_dir,
                      shp_path=shp_path, 
                      ortho_path=ortho_path,
                      xymax=xymax,
                      background="ortho", # options: "shp", "ortho", "both", "none"
                      bf_crs="EPSG:32627"
                      )

# Create a GIF animation of the saved png files
gif_name = f"{t_dt:%Y-%m-%d}_animation.gif"
gif_path = png_dir / gif_name
frames = [imageio.imread(png) for png in sorted(png_dir.glob("*.png"))]
imageio.mimsave(
    gif_path,
    frames,
    fps=1, # frames per second
    loop=0
)