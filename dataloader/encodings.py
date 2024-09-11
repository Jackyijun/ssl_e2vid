"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

import numpy as np
import torch


def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)


def events_to_mask(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into a binary mask.
    """

    device = xs.device
    img_size = list(sensor_size)
    mask = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    mask.index_put_((ys, xs), ps.abs(), accumulate=False)

    return mask


def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(180, 240)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)

def events_to_bilts(xs, ys, ts, framesize, t_range, num_bins=5, norm=True):
    """
    Calculate the global bidirectional time surface for each event:
    For each pixel in the entire CMOS, find the timestamp of the event that is closest to the current event on the time axis in the past.
    If there is no event in the past at this pixel, find the timestamp of the event that is closest to the current event on the time axis in the future.
    Args:
        xs: x coordinates of events, numpy array, shape: (N,)
        ys: y coordinates of events, numpy array, shape: (N,)
        ts: timestamps of events, a sorted numpy array, shape: (N,)
        framesize: the size of the CMOS, tuple, shape: (H, W)
        t_range: the time range of the time surface, int
        num_bins: the number of bins for the time surface, int
        norm: whether to normalize the time surface, bool
    Returns:
        time_surface: the local bidirectional time surface for the current event, numpy array, shape: (2*r+1, 2*r+1)
    """
    H, W = framesize
    
    # Array dimensions for the time surface
    time_surface = np.full((num_bins, H, W), -np.inf)  # Use np.inf as a placeholder for unset values
    
    # Calculate the relative positions of ts
    t_cur = 0
    
    
    # Get indices before and after the current event
    cur_idx = 0 # np.searchsorted(ts[prev_idx:], t_cur, side='right') + prev_idx
    past_indices = np.array([], dtype=int)
    
    for bidx in range(num_bins):
        # Array dimensions for the time surface of the current bin
        time_surface_bin = np.full((H, W), -np.inf)  # Use np.inf as a placeholder for unset values
        t_cur = t_cur.item() if isinstance(t_cur, torch.Tensor) else t_cur
        t_norm = ts - t_cur 
        # Get indices before and after the current event
        after_idx = np.searchsorted(ts[cur_idx:], t_cur + t_range, side='right') + cur_idx if bidx != num_bins - 1 else cur_idx
        future_indices = np.arange(cur_idx, after_idx) if bidx != num_bins - 1 else np.array([], dtype=int)
    
        # Update time_surface for past events (choose minimum time difference for each pixel)
        if len(past_indices) > 0:
            np.maximum.at(time_surface_bin, np.array([int(ys[past_indices][0]), int(xs[past_indices][0])]), t_norm[past_indices])

        # Temporary array to store future time differences, keeping only those cells that are still inf in time_surface
        if len(future_indices) > 0:
            future_time_surface_bin = np.full_like(time_surface_bin, np.inf)
            print("shape of future time surface bin ", future_time_surface_bin.shape)
            print("shape of middle ", np.array([int(ys[future_indices][0]), int(xs[future_indices][0])]).shape)
            print("shape of t norm ", t_norm[future_indices].shape)
            np.minimum.at(future_time_surface_bin, np.array([ys[future_indices], xs[future_indices]]), t_norm[future_indices])

        # Combine past and future times, only filling future times where past times were not updated
        mask = np.isinf(time_surface_bin)  # Find where past updates have not occurred
        if len(future_indices) > 0:
            time_surface_bin[mask] = future_time_surface_bin[mask]

        # Replace any remaining np.inf with -t_range (indicating no events found in either direction)
        time_surface_bin[np.isinf(time_surface_bin)] = -t_range
    
        # Fill in the missing pixels, disable for now
        # time_surface = bilinear_interpolation(time_surface)

        time_surface[bidx] = time_surface_bin
        t_cur += t_range
        cur_idx = after_idx
        past_indices = future_indices
    
    print("time_surface befoer norm ", time_surface)
    # Normalize to [-1, 1], while keep the empty cells as -1
    if norm:
        if t_range == torch.tensor(0.):
            t_range = torch.tensor(1)
        time_surface = time_surface / t_range
    
    print("t range is ", t_range)
    print(time_surface.float())
    return time_surface.float()
    # return torch.rand(5,128,128)


def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])


def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    """
    Returns binary mask to remove events from hot pixels.
    """

    mask = torch.ones(event_rate.shape).to(event_rate.device)
    if idx > min_obvs:
        for i in range(max_px):
            argmax = torch.argmax(event_rate)
            index = (argmax // event_rate.shape[1], argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask
