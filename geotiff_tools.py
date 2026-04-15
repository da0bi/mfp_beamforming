#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path

from pyproj import Transformer

import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds


def geotiff_crs_transform(geotiff_path, output_path, target_epsg="EPSG:32627"):
    """
    Reproject a GeoTIFF to another coordinate system.

    Parameters
    ----------
    geotiff_path : str or Path
        Input GeoTIFF file path.
    output_path : str or Path
        Output file path for reprojected GeoTIFF.
    target_epsg : str
        EPSG code for targeted coordinate system.
    """

    with rasterio.open(geotiff_path) as src:
        # get source CRS and transform
        src_crs = src.crs
        src_transform = src.transform

        # define target CRS (UTM)
        transformer = Transformer.from_crs(src_crs, target_epsg, always_xy=True)

        # read data
        data = src.read()

        # get bounds of the source raster
        left, bottom, right, top = src.bounds

        # transform bounds to target CRS
        left, bottom = transformer.transform(left, bottom)
        right, top = transformer.transform(right, top)

        # calculate new transform for the target CRS
        dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, src.width, src.height)

        # update metadata for the target CRS
        dst_meta = src.meta.copy()
        dst_meta.update(
            {
                "crs": target_epsg,
                "transform": dst_transform,
                "width": src.width,
                "height": src.height,
            }
        )

        # write output
        with rasterio.open(output_path, "w", **dst_meta) as dst:
            dst.write(data)

    return output_path


def merge_geotiffs(input_folder, output_path):
    """
    Merge all GeoTIFFs in a folder into a single GeoTIFF.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing GeoTIFFs.
    output_path : str or Path
        Output file path for merged GeoTIFF.
    """

    input_folder = Path(input_folder)

    # collect all tif files
    tif_files = sorted(input_folder.glob("*.tif"))

    if not tif_files:
        raise ValueError("No GeoTIFFs found in folder.")

    # open datasets
    src_files = [rasterio.open(fp) for fp in tif_files]

    # merge
    mosaic, out_transform = merge(src_files)

    # use metadata from first file
    out_meta = src_files[0].meta.copy()

    # update metadata
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0]
    })

    # write output
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # close all files
    for src in src_files:
        src.close()

    return output_path


def crop_orthofoto(
    ortho_path,
    xmin, xmax, ymin, ymax,
    output_path
):
    """
    Crop orthophoto to beamforming extent and save as new GeoTIFF.

    Parameters
    ----------
    ortho_path : str or Path
        Input GeoTIFF.
    xmin, xmax, ymin, ymax : float
        Cropping extent (must be in same CRS as raster!).
    output_path : str or Path
        Output file path for cropped GeoTIFF.

    Returns
    -------
    ortho : np.ndarray
        Cropped orthophoto (H, W, bands).
    extent : list
        [xmin, xmax, ymin, ymax]
    output_path : Path
        Path to saved cropped GeoTIFF.
    """

    ortho_path = Path(ortho_path)
    output_path = Path(output_path)

    with rasterio.open(ortho_path) as src:

        # Create window from bounds
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)

        # Read cropped data
        ortho = src.read(window=window)

        # 🚨 Check for empty result
        if ortho.shape[1] == 0 or ortho.shape[2] == 0:
            raise ValueError("Cropped raster is empty → check CRS / bounds!")

        # Get transform for cropped window
        transform = src.window_transform(window)

        # Copy and update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": ortho.shape[1],
            "width": ortho.shape[2],
            "transform": transform
        })

        # Save cropped GeoTIFF
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(ortho)

    # Convert to (H, W, bands) for plotting
    ortho = np.transpose(ortho, (1, 2, 0))

    # Define extent (for plotting)
    extent = [
        transform.c,
        transform.c + transform.a * ortho.shape[1],
        transform.f + transform.e * ortho.shape[0],
        transform.f
    ]

    return ortho, extent, output_path
