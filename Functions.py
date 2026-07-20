import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import label, find_objects, center_of_mass, generate_binary_structure, median_filter, gaussian_filter, uniform_filter
from scipy.interpolate import interp1d, RegularGridInterpolator
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
import pandas as pd
import datetime
import colormaps as cmaps
from skimage.filters import sato, gabor
from skimage.morphology import remove_small_objects
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pyart
import glob
import os
import plotly.graph_objects as go
import hdbscan
from scipy.signal import convolve2d
from scipy.stats import pearsonr
from skimage.morphology import closing, disk, skeletonize, opening
import hdbscan
from skimage.filters import apply_hysteresis_threshold
import time
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d

def find_frequency(ds, z):

    """
    This function finds the dominant radial frequency of waves within the residual velocity CAPPI. 
    The purpose is to find the dominant frequency of the field and plug it into the Gabor Filter later.

    Args:
        ds: This is an xarray ds containing all CAPPI data from the Radx2Grid conversion for a particular time.
        z: The CAPPI z0 level (km) in ds in which you want to find the radial frequency (ex: z = 0.35).

    Returns:
        dominant_feature_freq: a float value representing the dominant radial (spatial) frequency. 
                               The radial frequency is the magnitude of the frequency in the x and y direction.
        dominant_wavelength: a float value, inverse of dominant_feature_freq.
        bin_centers: numpy array representing the bins that radial frequency are split into based on maximum frequency.
        radial_power: numpy array representing mean power from the FFT power spectrum.
        
    """

    # Define the residual velocity field and its grid parameters
    r_vel = ds.sel(z0=z).isel(time=0).r_vel

    ny, nx = r_vel.shape

    dx = np.diff(r_vel.x0)[0]
    dy = np.diff(r_vel.y0)[0]

    Z = r_vel.values

    # Fill NaNs or else FFT will not work
    field_mean = np.nanmean(Z)
    Z_filled = np.nan_to_num(Z, nan=field_mean)

    # FFT
    F = np.fft.fft2(Z_filled)
    F_shifted = np.fft.fftshift(F)

    # Power spectrum
    power_spectrum = np.abs(F_shifted) ** 2

    # Remove DC (first point of frequency spectrum, represents average frequency of domain) component by setting it to 0
    cy, cx = power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2
    power_spectrum[cy, cx] = 0

    # Frequency axes
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    # Radial frequency grid
    Fx, Fy = np.meshgrid(freq_x, freq_y)
    radial_freq_grid = np.sqrt(Fx**2 + Fy**2) # Magnitude of x and y frequency

    # Splitting up radial frequency into bins
    max_freq = np.max(radial_freq_grid)
    num_bins = min(power_spectrum.shape) // 2
    bins = np.linspace(0, max_freq, num_bins)

    bin_indices = np.digitize(radial_freq_grid, bins)

    # Radially averaged power
    radial_power = np.zeros(len(bins) - 1)

    for i in range(1, len(bins)):
        mask = (bin_indices == i)

        if np.any(mask):
            radial_power[i - 1] = np.mean(power_spectrum[mask])

    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Remove DC contribution
    radial_power[0] = 0

    # Dominant frequency
    dominant_idx = np.argmax(radial_power)

    dominant_feature_freq = bin_centers[dominant_idx]
    dominant_wavelength = 1.0 / dominant_feature_freq

    # Optional print statements 
    # print(f"Dominant Feature Frequency: {dominant_feature_freq:.4f} cycles/unit")
    # print(f"Dominant Feature Spacing (Wavelength): {dominant_wavelength:.4f} units")

    return dominant_feature_freq, dominant_wavelength, bin_centers, radial_power

def subsetFilter(ds, vad, z, smooth_size = 6, filter_threshold = 0.25):

    """
    This function applies an image filter to the CAPPI residual velocity data.
    The purpose is remove background noise outside of rolls.

    Args:
        ds: This is an xarray ds containing all CAPPI data from the Radx2Grid conversion for a particular time.
        vad: This is the xarray dataset containing the VAD data for a particular time.
        z: The CAPPI z0 level (km) in ds that you are analzying.
        smooth_size: The size parameter for the scipy.ndimage.median_filter function. 
                     Represents the every nth value taken as input into the filter function. 
                     (ex: smooth_size = 6, every 6th point is taken to be smoothed. A 6-point rolling median across a 2D field).           
        filter_threshold: A float value representing the minimum strength of residual velocity across the filtered field.
                          Every values > filter_threshold is saved, the rest are set to 0.

    Returns:
        subset_filled: A 2D array representing the residual velocity field, except all NaNs (radar hole) are now 0.
        mean_vad_direction: The mean direction of the VAD wind from the surface layer height to the jet height.
        ref_filled: reflectivity from the CAPPI where NaNs are filled as 0.
        strong_filtered: A 2D array representing the filtered residual velocity
        
    """

    # Grabbing residual velocity subset and filling in NaNs
    subset = ds.sel(z0 = z).isel(time = 0).r_vel
    subset_filled = np.nan_to_num(subset.values, 0)

    # Doing the same for reflectivity
    ref = ds.sel(z0 = z).isel(time = 0).REF
    ref_filled = np.nan_to_num(ref.values, 0)

    # Applying the median_filter to the filled residual velocity field
    filtered = median_filter(subset_filled, size=smooth_size)
    strong_filtered = np.where(np.abs(filtered) > filter_threshold, filtered, 0)

    # Finding the mean VAD direction within below the VAD jet height
    direction = vad.direction_12swp
    speed = vad.speed_12swp.values
    heights = vad.height.values
    jet_height = heights[np.nanargmax(speed)]
    sfc_layer_height = heights[np.nanargmin(np.abs(heights - 0.15*jet_height))]
    mean_vad_direction = (270 - np.nanmean(direction.sel(height = slice(sfc_layer_height, jet_height)).values) + 90)%360
    
    return subset_filled, mean_vad_direction, ref_filled, strong_filtered

def GaborMask(strong_filtered, theta, frequency=0.25, min_object_size = 10, gaussian_sigma=2):
    
    """
    This function applies a Gabor filter mask to isolate and segment wave-like ridge features 
    (e.g., horizontal convective rolls) from a pre-filtered residual velocity field. 
    It leverages the directional and frequency-selective nature of the 2D Gabor filter, defined mathematically as:

    g(x, y; lambda, theta, psi, sigma, gamma) = exp(-(x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * exp(i * (2 * pi * x' / lambda + psi))

    Where:
        x' = x * cos(theta) + y * sin(theta)
        y' = -x * sin(theta) + y * cos(theta)
        lambda = 1 / frequency (spatial wavelength)
        psi = 0 (phase offset)
        gamma = 1 (spatial aspect ratio)

    The orientation angle (theta) is represented by the mean VAD wind direction and is converted from degrees to radians within the function.
    
    Args:
        strong_filtered: A 2D numpy array representing the pre-filtered residual velocity field. Output from subsetFilter function.
        theta: A float value representing the orientation angle of the Gabor filter in degrees. This is the mean VAD direction from the subsetFilter function.
        frequency: A float value representing the spatial frequency of the sinusoidal factor (default is 0.25). This is obtained from the output of the find_frequency function.
        min_object_size: An integer representing the minimum size (in pixels) of valid objects to retain in the mask.
        gaussian_sigma: A float value representing the standard deviation for the initial Gaussian smoothing kernel.

    Returns:
        labeled_arms: A 2D integer array where detected coherent wave structures are uniquely labeled.
        filt_real: A 2D array containing the real component of the Gabor-filtered field highlighting structural ridges.
        input_field: A 2D array representing the magnitude of the strong_filtered input after Gaussian smoothing.
        stdev: A float representing the standard deviation of the smoothed input field magnitude.
        clean_mask: A 2D boolean array representing the thresholded and size-filtered binary wave mask.
        
    """

    # Applying a gaussian filter to the pre-filtered residual velocity field (output of subsetFilter).
    # The gabor output results in a real and an imaginary part - see Euler's formula. We use the real component.
    input_field = gaussian_filter(np.abs(strong_filtered), sigma=gaussian_sigma)
    filt_real, filt_imag = gabor(input_field, frequency=frequency, theta=np.deg2rad(theta)) 

    # Finding only the strong wave-like features outputted from the Gabor filter (possible rolls) 
    ridges = np.abs(filt_real)
    ridges_threshold = np.percentile(filt_real, 95) # Strong is > 95th percentile

    # Creating a mask where potential roll features could be
    wave_mask = (ridges > ridges_threshold) & (np.abs(strong_filtered) > 0.25)

    clean_mask = remove_small_objects(wave_mask, min_size=min_object_size)

    labeled_arms, num_arms = label(clean_mask)

    stdev = np.std(input_field)

    props = regionprops(labeled_arms)

    labeled_arms, num_arms = label(clean_mask)
    
    return labeled_arms, filt_real, input_field, stdev, clean_mask

def feature_DBSCAN(ds, labeled_arms, z, eps = 0.9):

    """
    This function groups potential roll-like features together into clusters using the DBSCAN algorithm.
    
    Args:
        ds: This is an xarray ds containing all CAPPI data from the Radx2Grid conversion for a particular time.
        labeled_arms:  A 2D integer array where detected coherent wave structures are uniquely labeled. Output from the GaborMask function.
        z: The CAPPI z0 level (km) in ds that you are analzying.
        eps: float parameter that describes strictness of the DBSCAN algorithm. The radius of the nth dimensional shape used in clustering features.

    Returns:
        df_features: A pandas dataframe consisting of properties of each feature and its cluster id. 
        
    """

    subset = ds.sel(z0 = z).isel(time = 0)
    r_vel = subset.r_vel.values

    # Gathering features/objects and their locations

    props = regionprops(labeled_arms, intensity_image=r_vel)

    feature_data = []
    
    for prop in props:

        # Get the centroid (center of mass) of the feature
        y_idx, x_idx = prop.centroid

        # Map back to physical coordinates
        y_loc = subset.y0.values[int(y_idx)]
        x_loc = subset.x0.values[int(x_idx)]

        blob_pixels = prop.intensity_image[prop.image]
        velocity_range = np.percentile(blob_pixels, 95) - np.percentile(blob_pixels, 5)
        minor_axis = prop.minor_axis_length
        shear_proxy = velocity_range / minor_axis if minor_axis > 0 else 0
        
        # Angle of the major axis of the ellipse that fits the feature
        angle_rad = prop.orientation
        angle_deg = np.degrees(angle_rad)
        major_axis = prop.major_axis_length
        minor_axis = prop.minor_axis_length
        eccentricity = prop.eccentricity
        
        feature_data.append({
            'label_id': prop.label,
            'x': x_loc,
            'y': y_loc,
            'angle_deg':angle_deg,
            'major_axis':major_axis,
            'minor_axis':minor_axis,
            'eccentricity':eccentricity,
            'r_vel_centroid':shear_proxy
        })
    
    # Convert to a DataFrame
    df_features = pd.DataFrame(feature_data)

    if df_features.empty == True:
        return
    
    # Clustering features together

    df_features['angle_rad'] = np.radians(df_features['angle_deg'])
    df_features['cos_2theta'] = np.cos(2 * df_features['angle_rad'])
    df_features['sin_2theta'] = np.sin(2 * df_features['angle_rad'])
    
    # Define the traits we want to group by: Location, angle, residual velocity shear across roll
    clustering_features = df_features[['x', 'y', 'angle_rad', 'r_vel_centroid']] # 1   
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_features)

    # Apply weights to each feature
    weights = [1, 1, 0.9, 0.3]
    scaled_data = scaled_data*weights

    # Cluster the features using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=2)
    
    # Assign a new ID to group these
    df_features['super_group_id'] = dbscan.fit_predict(scaled_data)
    
    # Calculate stdev only where there is actual signal activity
    active_signal = input_field[input_field > 0.25]
    active_stdev = np.std(active_signal) if active_signal.size > 0 else 0
    
    if active_stdev < 0.15:
        df_features['super_group_id'] = [-1 for i in range(len(df_features['super_group_id'].values))]

    return df_features

# The following two functions are helper functions used in making cross sections

def ccw(A, B, C):

    """Check if three points are listed in a counter-clockwise order."""

    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):

    """Return True if line segment AB intersects with line segment CD."""

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def makeCrossSections(ds, df_features, mean_vad_direction):

    """
    This function uses the CAPPI data from Radx2Grid and detected features to create cross sections across potential roll features.

    Args:
        ds: This is an xarray ds containing all CAPPI data from the Radx2Grid conversion for a particular time.
        df_features: A pandas dataframe consisting of properties of each feature and its cluster id. Output from feature_DBSCAN.
        mean_vad_direction: The mean VAD wind direction below the jet height. Output from subsetFilter.

    Returns:
        cross_sections: List of dictionaries containing data and information for each cross sections.
    """

    # Gathering coordinates from ds
    x0 = ds.x0.values
    y0 = ds.y0.values
    z0 = ds.z0.values
    
    # Averaging the feature groups
    num_groups = df_features.super_group_id.values.max()
    groups = [df_features[df_features.super_group_id == i] for i in range(0, num_groups+1, 1)]
    orientations_dg = [(270 - group.mean().angle_deg + 90)%360 for group in groups] # mean orientation of each cluster of roll-like features
    
    # Cross section directions
    orientations_rad = np.deg2rad(orientations_dg)
    
    # Creating bounding boxes for the cross sections
    boxes = []
    
    for gid, group in df_features.groupby("super_group_id"):
        if gid == -1:
            continue
    
        box = {
            "gid": gid,
            "xmin": group.x.min(),
            "xmax": group.x.max(),
            "ymin": group.y.min(),
            "ymax": group.y.max(),
            'minor_axis':group.minor_axis,
        }
    
        boxes.append(box)
    
    # Number of points for cross section line
    num_points = 50
    cross_sections = []


    for i, box in enumerate(boxes):
    
        theta = orientations_rad[i]

        # Compute smallest angular difference from VAD direciton (0–90° effectively)
        angle_diff = np.abs(orientations_dg[i] - mean_vad_direction)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        angle_diff = np.minimum(angle_diff, 180 - angle_diff)

        # Based on Morrison et al. (2005) roll orientation distribution
        # If angle difference between mean_vad_direction and mean orientation of feature group is too large
        if angle_diff > 50:
            continue

        # If box surrounding group of features is too big (greater than 19 km square) get rid of it
        xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']

        if np.abs(xmax-xmin) > 19 or np.abs(ymax-ymin) > 19:
            continue

        # Math to determine center of feature box
    
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
    
        # Direction vector 
        t = np.array([np.cos(theta), np.sin(theta)])
    
        # Normal vector 
        n = np.array([-t[1], t[0]])
    
        corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax]])

        corners_relative = corners - np.array([cx, cy])
    
        # Projection
        proj = corners_relative @ t
        s_min, s_max = proj.min(), proj.max()

        s_vals = []

        if t[0] != 0:
            s_vals += [(xmin - cx)/t[0], (xmax - cx)/t[0]]
        if t[1] != 0:
            s_vals += [(ymin - cy)/t[1], (ymax - cy)/t[1]]
        
        points = [(cx + s*t[0], cy + s*t[1], s) for s in s_vals]
        
        # Keep only points inside the box
        valid = [(x,y,s) for x,y,s in points if xmin <= x <= xmax and ymin <= y <= ymax]
        
        # Extract valid s range
        s_valid = [p[2] for p in valid]
        if len(s_valid) == 0:
            continue
        s_min, s_max = min(s_valid), max(s_valid)

        extension = np.percentile(box['minor_axis'], 75) / 2
        s_min -= extension
        s_max += extension
        
        # Clamp s_min and s_max to the Domain Bounding Box
        domain_xmin, domain_xmax = x0.min(), x0.max()
        domain_ymin, domain_ymax = y0.min(), y0.max()

        # Find where the mathematical line intersects the domain boundaries
        s_domain = []
        if t[0] != 0:
            s_domain += [(domain_xmin - cx)/t[0], (domain_xmax - cx)/t[0]]
        if t[1] != 0:
            s_domain += [(domain_ymin - cy)/t[1], (domain_ymax - cy)/t[1]]

        # Filter to true domain intersections
        valid_domain_s = [
            sd for sd in s_domain 
            if domain_xmin - 1e-4 <= (cx + sd*t[0]) <= domain_xmax + 1e-4 
            and domain_ymin - 1e-4 <= (cy + sd*t[1]) <= domain_ymax + 1e-4
        ]
        
        # Clamp your extended s limits so they don't exceed the domain
        if valid_domain_s:
            max_s_min, max_s_max = min(valid_domain_s), max(valid_domain_s)
            s_min = max(s_min, max_s_min)
            s_max = min(s_max, max_s_max)
        # -------------------------------------------------------------

        # Line along the cross section direction
        s = np.linspace(s_min, s_max, num_points)
        x_cross = cx + s * t[0]
        y_cross = cy + s * t[1]
            
        # Interpolation function for actual cross sections
        interp_func = RegularGridInterpolator(
                (z0, y0, x0),
                ds.isel(time = 0).r_vel.values,
                bounds_error=False,
                fill_value=np.nan
            )


        cross_section = np.array([
                interp_func((z, y_cross, x_cross)) for z in z0
        ])

        ref_height = np.argwhere(z0 == z)
        if np.nanmax(cross_section[ref_height]) < 1.75:
            continue
        

        # print(np.nanmax(np.abs(cross_section[0:5])))
        if np.nanmax(np.abs(cross_section[0:5])) < 2:
            continue
    
    
        cross_sections.append({
        "gid": box["gid"],
        "x": x_cross,
        "y": y_cross,
        "s":s,
        "profile": cross_section})

    filtered_cross_sections = []

    # Making sure cross sections don't overlap
    for cs in reversed(cross_sections):
        A = (cs["x"][0], cs["y"][0])
        B = (cs["x"][-1], cs["y"][-1])
        
        has_overlap = False
        
        for kept_cs in filtered_cross_sections:
            C = (kept_cs["x"][0], kept_cs["y"][0])
            D = (kept_cs["x"][-1], kept_cs["y"][-1])
            
            if segments_intersect(A, B, C, D):
                has_overlap = True
                break 
                
        if not has_overlap:
            filtered_cross_sections.append(cs)
            
    cross_sections = filtered_cross_sections[::-1]

    return cross_sections

def gradient_label(
    r_vel, 
    heights, 
    distances, 
    grad_thresh=0.3,         
    strong_thresh=1.25,       
    weak_thresh=0.8,         
    ground_height_max=0.2,  
    grad_height_max=1.0,
    neighbor_threshold=3.0,  
    min_depth_km=0.25,        
    max_width_km=6.0         
):
    """
    Identifies, validates, and pairs ground-reaching hurricane boundary layer rolls in cross sections.

    Args:
        r_vel: 2D numpy array containing the residual velocity data from a cross section.
        heights: Height dimension of CAPPIs of residual velocity data.
        distances: Array containing distances along the cross section direction. "s" from makeCrossSections.
        grad_thresh: The minimum of the normalized gradient of residual velocity in order for roll detection.
        strong_thresh: Any residual velocity below this float is discounted
        weak_thresh: The threshold of residual velocity for the outer edge of a roll
        ground_height_max: The bottom of the roll must be at or below this height (km)
        grad_height_max: The gradient connecting positive and negative residual velocity features must be at or below this height (km)
        neighbor_threshold: The maximum distance between positive and negative residual velocity pairs (km)
        min_depth_km: The minimum depth that a roll feature can have (km)
        max_width_km: The maximum wavelength for a roll feature (km)

    Returns:
        r_vel_masked: The data with the residual velocity positive and negative pair filtered out. Rest is NaN.
        df_matches: Pandas dataframe containing pairs of residual velocity features (rolls) sorted by distance between pairs.
    """
    struct = generate_binary_structure(rank=2, connectivity=1)

    # Gradient Setup 
    dz, dx = np.gradient(r_vel)
    gmag = np.sqrt(dz**2 + dx**2)
    gmax = np.nanpercentile(gmag, 98) 
    mask_grad = (gmag / gmax) > grad_thresh

    # Positive and negative residual velocity masks
    mask_pos = r_vel > strong_thresh
    mask_neg = r_vel < -strong_thresh
    
    # Label strong gradients between positive and negative residual velocity features (where rolls are)
    lbl_grad, _ = label(mask_grad, structure=struct)
    # Label positive and negative regions of residual velocity
    lbl_pos, n_pos = label(mask_pos, structure=struct)
    lbl_neg, _ = label(mask_neg, structure=struct)

    lbl_res = lbl_pos + np.where(lbl_neg > 0, lbl_neg + n_pos, 0)

    # Find strong residual velocity objects (this is a positive or negative residual velocity blob in a roll pair)
    objs_grad_raw = find_objects(lbl_grad)
    objs_res_raw = find_objects(lbl_res)

    # Helper functions
    def get_height(y_idx, x_idx):
        return heights[y_idx] if heights.ndim == 1 else heights[y_idx, x_idx]

    # Make sure the residual velocity objects touch the ground
    def touches_local_ground(y_slice, x_slice):
        lowest_valid_idx = 0
        return y_slice.start <= (lowest_valid_idx + 2)

    # Filter Gradient Objects
    valid_gradient_objs = []
    for i, sl in enumerate(objs_grad_raw):
        if sl is None: continue
        cx = int((sl[1].start + sl[1].stop) / 2)
        bot_h = get_height(sl[0].start, cx)
        
        # Gradient is allowed aloft, ignoring local ground checks
        if bot_h <= grad_height_max:
            valid_gradient_objs.append((i + 1, sl, lbl_grad[sl] == (i + 1)))

    # Filter Residual Velocity Objects
    all_residual_objs = [] # Add filtered residual velocity object dictionaries to a list
    for i, sl in enumerate(objs_res_raw):
        if sl is None: continue 
        
        y_slice, x_slice = sl
        cx = int((x_slice.start + x_slice.stop) / 2)
        
        bot_h = get_height(y_slice.start, cx)
        top_h = get_height(y_slice.stop - 1, cx)
        
        obj_depth = top_h - bot_h
        obj_width = abs(distances[x_slice.stop - 1] - distances[x_slice.start])

        # Velocity cores MUST touch ground
        is_grounded = touches_local_ground(y_slice, x_slice) and (bot_h <= ground_height_max)

        if is_grounded and (obj_depth >= min_depth_km) and (obj_width <= max_width_km):
            all_residual_objs.append({ # Each object has an id, slices (sections of data), and a mask
                "id": i + 1, 
                "slice": sl, 
                "mask": lbl_res[sl] == (i + 1)
            })

    # Determine "Primary" (Gradient-Matched)
    for r_obj in all_residual_objs:

        # Grab the slice and mask from residual velocity object
        r_slice, r_mask_local = r_obj['slice'], r_obj['mask']
        # Create an empty array like the residual velocity shape then overlay each object mask on it
        r_mask_full = np.zeros_like(r_vel, dtype=bool)
        r_mask_full[r_slice] = r_mask_local

        # Find the sign of that residual velocity object and find its center
        cy, cx = np.mean(np.where(r_mask_full), axis=1)
        r_obj['sign'] = 1 if np.mean(r_vel[r_mask_full]) > 0 else -1
        r_obj['center_dist_km'] = distances[int(cx)]
        r_obj['is_primary'] = False 

        # Each residual velocity object is set as the primary object in a pair if the residual velocity object and gradient object overlap
        for _, g_slice, g_mask_local in valid_gradient_objs:
            g_mask_full = np.zeros_like(r_vel, dtype=bool)
            g_mask_full[g_slice] = g_mask_local
            if np.any(g_mask_full & r_mask_full):
                r_obj['is_primary'] = True
                break 

    # Pairing Logic

    # Sort residual velocity objects by distance between them
    all_residual_objs.sort(key=lambda x: x['center_dist_km'])
    paired_ids = set()
    final_valid_objects = []

    for obj in all_residual_objs:
        if not obj['is_primary'] or obj['id'] in paired_ids: 
            continue

        # Finding the best partner for the primary residual velocity object
        best_partner = None
        min_dist = float('inf')

        for partner in all_residual_objs:

            # Getting rid of the residual velocity object if its already pair or has the same sign as the one closest to it
            if partner['id'] == obj['id'] or partner['id'] in paired_ids or partner['sign'] == obj['sign']: 
                continue

            dist = abs(obj['center_dist_km'] - partner['center_dist_km'])

            # Make sure the objects are close enough together (see docstring)
            if dist < neighbor_threshold and dist < min_dist:
                min_dist = dist
                best_partner = partner
        
        if best_partner:
            paired_ids.update([obj['id'], best_partner['id']])
            pair = sorted([obj, best_partner], key=lambda x: x['center_dist_km'])
            final_valid_objects.extend(pair) # Add to existing list of pairs

    # Object Expansion & Final Masking

    # Make sure we are labeling the WHOLE roll not just the strong part of the residual velocity object as above
    lbl_pos_weak, _ = label(r_vel >= weak_thresh, structure=struct)
    lbl_neg_weak, _ = label(r_vel <= -weak_thresh, structure=struct)

    # Creating empty mask as a canvas to add strong and weak residual velocity object masks
    final_residual_mask = np.zeros_like(r_vel, dtype=bool)
    matches, processed_final_ids = [], set()

    for obj in final_valid_objects:
        strong_mask_full = np.zeros_like(r_vel, dtype=bool)
        strong_mask_full[obj['slice']] = obj['mask']

        weak_lbl_map = lbl_pos_weak if obj['sign'] == 1 else lbl_neg_weak
        weak_ids = np.unique(weak_lbl_map[strong_mask_full])
        weak_ids = weak_ids[weak_ids > 0]

        expanded_mask = np.zeros_like(r_vel, dtype=bool)
        if len(weak_ids) > 0:
            for w_id in weak_ids:
                expanded_mask |= (weak_lbl_map == w_id)
        else:
            expanded_mask = strong_mask_full

        # Adding expanded residual velocity object to empty mask
        final_residual_mask |= expanded_mask

        if obj['id'] not in processed_final_ids:
            y_indices, x_indices = np.where(expanded_mask)
            
            if heights.ndim == 1:
                obj_top, obj_bot = heights[np.max(y_indices)], heights[np.min(y_indices)]
            else:
                obj_top = heights[np.max(y_indices), x_indices[np.argmax(y_indices)]]
                obj_bot = heights[np.min(y_indices), x_indices[np.argmin(y_indices)]]

            # Pairing matched extended residual velocity objects into rolls
            matches.append({
                "Residual_ID": obj['id'],
                "Sign": "Positive" if obj['sign'] == 1 else "Negative",
                "Center_Dist_km": obj['center_dist_km'], 
                "Is_Gradient_Matched": obj['is_primary'],
                "Top_Height": obj_top,
                "Bottom_Height": obj_bot
            })
            processed_final_ids.add(obj['id'])

    r_vel_masked = np.where(final_residual_mask, r_vel, np.nan)
    df_matches = pd.DataFrame(matches).sort_values(by="Center_Dist_km") if matches else pd.DataFrame()

    return r_vel_masked, df_matches

def calculate_half_wavelength_profiles(r_vel_masked, heights, distances, df_matches):
    """
    Calculates the half-wavelength profiles at every height level for identified roll pairs.
    This function processes a 2D vertical cross-section of residual velocity data to track 
    the spatial wavelength and geometric properties of paired roll features across overlapping heights.

    Args:
        r_vel_masked: A 2D numpy array representing the masked residual velocity field.
        heights: A 1D or 2D numpy array containing the vertical height levels (km).
        distances: A 1D numpy array containing distances along the cross section direction. "s" from makeCrossSections.
        df_matches: Pandas dataframe containing pairs of residual velocity features (rolls) sorted by distance between pairs.

    Returns:
        profiles: A dictionary where each key is a string identifier for a roll pair (e.g., 'Pair_1') 
                  and the value is a pandas DataFrame tracking wavelength metrics dynamically by height.
        
    """
    # Early exit if no valid roll matches were passed to the function
    if df_matches.empty:
        return {}

    # Define a 2D connectivity structure for component labeling
    struct = generate_binary_structure(rank=2, connectivity=1)
    
    # Separately label the positive and negative velocity patches to keep sign bounds distinct
    lbl_pos, n_pos = label(r_vel_masked > 0, structure=struct)
    lbl_neg, _ = label(r_vel_masked < 0, structure=struct)
    
    # Combine the labels into a single array, shifting negative labels to prevent ID conflicts
    lbl_mask = lbl_pos + np.where(lbl_neg > 0, lbl_neg + n_pos, 0)
    
    # Map physical horizontal distances to their corresponding integer label IDs using center of mass
    dist_to_label = {}
    for i, sl in enumerate(find_objects(lbl_mask)):
        if sl is None: continue
        cy, cx = center_of_mass(lbl_mask == (i + 1))
        # Convert pixel x-coordinate (cx) to its physical kilometer distance equivalent
        phys_x = np.interp(cx, np.arange(len(distances)), distances)
        dist_to_label[phys_x] = i + 1

    # Helper function to find which labeled object matches the tracked physical distance column
    def get_label_id(target_dist):
        if not dist_to_label: return None
        closest_dist = min(dist_to_label.keys(), key=lambda x: abs(x - target_dist))
        # Enforce a strict spatial matching tolerance of 0.5 km
        return dist_to_label[closest_dist] if abs(closest_dist - target_dist) <= 0.5 else None

    profiles = {}
    # Iterate through the matched DataFrame in steps of 2 (handling object pairs)
    num_pairs = len(df_matches) // 2

    for i in range(num_pairs):
        obj_a, obj_b = df_matches.iloc[2*i], df_matches.iloc[2*i + 1]
        lid_a, lid_b = get_label_id(obj_a['Center_Dist_km']), get_label_id(obj_b['Center_Dist_km'])

        # Both structures must exist in the grid to calculate a spatial profile
        if lid_a is None or lid_b is None: continue

        # Generate boolean masks for each individual object in the pair
        mask_a, mask_b = (lbl_mask == lid_a), (lbl_mask == lid_b)
        y_a, y_b = np.where(mask_a)[0], np.where(mask_b)[0]
        
        if len(y_a) == 0 or len(y_b) == 0: continue

        # Isolate the core velocity values within each object mask and extract the maximum intensity
        rvel_mask_a = np.where(mask_a, r_vel_masked, np.nan)
        rvel_mask_b = np.where(mask_b, r_vel_masked, np.nan)
        obj_a_max, obj_b_max = np.nanmax(rvel_mask_a), np.nanmax(rvel_mask_b)

        # Determine the overlapping vertical height boundaries shared by both structures
        min_h_idx, max_h_idx = max(np.min(y_a), np.min(y_b)), min(np.max(y_a), np.max(y_b))
        pair_profile = []

        # Step through the cross-section line-by-line within the overlapping height band
        for h_idx in range(min_h_idx, max_h_idx + 1):
            row_a, row_b = mask_a[h_idx, :], mask_b[h_idx, :]
            # Both features must co-exist at this specific vertical row to calculate wavelength
            if not np.any(row_a) or not np.any(row_b): continue

            # Locate the absolute peak intensity coordinate within the combined pair bounding box
            pair_rvel = np.where(mask_a | mask_b, np.abs(r_vel_masked), np.nan)
            
            if np.all(np.isnan(pair_rvel)):
                max_height_km = np.nan
            else:
                max_h_idx_val = np.unravel_index(np.nanargmax(pair_rvel), pair_rvel.shape)[0]
                max_height_km = heights[max_h_idx_val] if heights.ndim == 1 else heights[max_h_idx_val, 0]

            # Calculate the mean horizontal pixel index for each feature on this row and map to km
            phys_x_a = np.interp(np.mean(np.where(row_a)[0]), np.arange(len(distances)), distances)
            phys_x_b = np.interp(np.mean(np.where(row_b)[0]), np.arange(len(distances)), distances)

            # Half-wavelength is defined as the absolute physical distance separating the two centers
            half_lambda = abs(phys_x_a - phys_x_b)
            # Filter out non-physical or half wavelengths below 100 meters
            if half_lambda < 0.1: continue

            # Compile parameters into a dictionary and append to a list
            pair_profile.append({
                "Height_km": heights[h_idx] if heights.ndim == 1 else heights[h_idx, 0],
                "Half_Wavelength_km": half_lambda,
                "Full_Wavelength_km": half_lambda * 2,
                "Obj_A_Center": phys_x_a,
                "Obj_B_Center": phys_x_b,
                "Top_Height_km": obj_a['Top_Height'],
                "Bottom_Height_km": obj_a['Bottom_Height'],
                "Max_Height_km": max_height_km,
                "Obj_A_Max": obj_a_max,
                "Obj_B_Max": obj_b_max
            })

        if pair_profile:
            profiles[f"Pair_{i+1}"] = pd.DataFrame(pair_profile)

    return profiles