#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shelve
import numpy as np
import json
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
    mean_bf_array, semb_max, semb_min, maskoutside,
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
    maskoutside : list or None
        List of [min, max] semblance values to mask the beamformer
        array outside of.
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

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

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

        if maskoutside is not None:
            semb_min = maskoutside[0]
            semb_max = maskoutside[1]

            masked_bf = np.ma.masked_outside(
                mean_bf_array,
                semb_min, semb_max
            )

        else:

            masked_bf = np.ma.masked_where(
                mean_bf_array < semb_min,
                mean_bf_array
            )


    else:

        masked_bf = mean_bf_array

    # -------------------------------
    # Plot beamformer
    # -------------------------------
    im = ax.imshow(
        masked_bf.T,  # transpose so (y, x)
        origin="lower",
        extent=[xcoord.min(), xcoord.max(), ycoord.min(), ycoord.max()],
        cmap="inferno",
        alpha=0.3 if background in ("ortho", "both") and ortho_path is not None else 1.0,
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
    plt.savefig(png_path, dpi=100)
    plt.close(fig)


#--------------------------------
# get parameters from config json
#--------------------------------
# define path to config json
#config_path = Path("/home/db/Software/psysmon/mfp_beamforming/beamform_plot.json")
config_path = Path(
    "/home/db/Projects/APO_Monitoring_2023/10_psysmon/output/"
    "mfp_beamforming/202306_09_run1/202306_09_run1_plot.json"
)

# get parameters
with config_path.open("r") as f:
    config = json.load(f)

directory = Path(config["directory"])
scoord = np.array(config["scoord"])
shp_path = config["shp_path"]
ortho_path = config["ortho_path"]
ortho_folder = Path(config["ortho_folder"])
semb_min = config["sembrng"][0]
semb_max = config["sembrng"][1]
maskoutside = config["maskoutside"]
plot_xymax = config["plot_xymax"]
background = config["background"]
bf_crs = config["bf_crs"]
fps_gif = config["fps_gif"]
fps_video = config["fps_video"]
plot_all_new = config["plot_all_new"]

# get all orthofoto files and dates if a folder of orthofotos is given
if background in ("ortho", "both") and ortho_folder is not None:
    all_orthofotos = list(ortho_folder.glob("*.tif"))
    orthofoto_dates = [(datetime.strptime(f.name[:8], '%Y%m%d').date(), f) for f in all_orthofotos]

#------------------------------------------------------------
# Loop through the subfolders in the directory,
# find *.db-files in the subfolders,
# load the data,
# compute and plot the hourly mean beamformer array,
# then create one GIF animation per day of the saved hourly png files,
# and a video for the entire period.
#------------------------------------------------------------
for subfolder in sorted(directory.iterdir()):

    if not subfolder.is_dir():

        continue

    if not plot_all_new:

        gif_folder = subfolder / "gif_frames"

        # Check if 'gif_frames' exists inside this subfolder
        if gif_folder.is_dir():

            # Count only .png files inside 'gif_frames'
            png_count = len(list(gif_folder.glob("*.png")))

            # If there are 24 pngs skip loop
            if png_count == 24:

                continue

            # if pngs are not 24 print the parent subfolder and skip loop
            print(f"Skipping {subfolder.name}: found {png_count} PNGs (expected 24)")

            continue

    # get orthofoto of the date closest to the beamforming results subfolder
    if background in ("ortho", "both") and ortho_folder is not None:

        # get the target date from subfolder (YYYY_DOY)
        target_date = datetime.strptime(subfolder.name, '%Y_%j').date()

        # find orthofoto with the date where the absolute difference in days is smallest
        # this handles both exact matches (diff=0) and closest dates
        closest_date, ortho_path = min(orthofoto_dates, key=lambda x: abs(x[0] - target_date))

    # take random timestamp and extract the x and y coordinates
    t = list(imp_shelve_file(subfolder, sorted(subfolder.glob("*.db"))[0].name).keys())[0]
    xcoord = imp_shelve_file(subfolder, sorted(subfolder.glob("*.db"))[0].name)[t]['xcoord']
    ycoord = imp_shelve_file(subfolder, sorted(subfolder.glob("*.db"))[0].name)[t]['ycoord']

    for fpath in sorted(subfolder.iterdir()):

        if fpath.is_file() and fpath.suffix == ".db":

            print(f"\n{fpath}")

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
            png_dir = fpath.parent / "gif_frames"
            png_dir.mkdir(exist_ok=True)

            plot_bf_array(mean_bf_array, semb_max, semb_min, maskoutside,
                          xcoord, ycoord, scoord,
                          title, png_name, png_dir,
                          shp_path=shp_path,
                          ortho_path=ortho_path,
                          xymax=xymax if plot_xymax else None,
                          background=background, # options: "shp", "ortho", "both", "none"
                          bf_crs=bf_crs
                          )

    # Create a GIF animation of the saved png files
    gif_name = f"{t_dt:%Y-%m-%d}_animation.gif"
    gif_path = png_dir / gif_name
    frames = [imageio.imread(png) for png in sorted(png_dir.glob("*.png"))]
    imageio.mimsave(
        gif_path,
        frames,
        fps=fps_gif, # frames per second
        loop=0
    )

# Create a video for the entire period
video_output_path = directory.parent / f"{directory.parent.name}.mp4"

all_pngs = sorted(directory.rglob("*.png"))

with imageio.get_writer(
    video_output_path,
    format="FFMPEG",
    mode="I",
    fps=fps_video,
    macro_block_size=1
) as writer:
    for png in all_pngs:
        image = imageio.imread(png)
        writer.append_data(image)
