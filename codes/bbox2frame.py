import numpy as np
from .utils import task
from . import vad
from scipy.ndimage import convolve


def simple_maximum(scores_bbox):
    # [T, M]: T: num of frame, M: num of bbox in each frame.
    T = len(scores_bbox)
    ret = []
    for t in range(T):
        arr = scores_bbox[t]
        if len(arr):
            ret.append(np.max(arr))
        else:
            ret.append(0)

    return np.asarray(ret)


def spatial(scores_bbox, dataset_name, bbox_meta, B=4, smooth_T=5, smooth_d=1):
    H, W = vad.get_shape(dataset_name)
    N_frames = len(scores_bbox)
    anomalymaps = np.zeros((N_frames, H // B, W // B), dtype=np.float32)

    for t in range(N_frames):
        scores = scores_bbox[t]
        bmetas = bbox_meta[t]
        M = len(scores)

        for m in range(M):
            score = scores[m]
            x1, y1, x2, y2 = bmetas[m]
            y1, y2, x1, x2 = y1 // B, y2 // B, x1 // B, x2 // B
            y1 = int(y1)
            y2 = int(y2)
            x1 = int(x1)
            x2 = int(x2)

            anomalymaps[t, y1: y2, x1: x2] = np.maximum(anomalymaps[t, y1:y2, x1: x2], score)

    # perform smoothing

    if smooth_T > 1 or smooth_d > 1:
        with task('Smooth'):
            d = smooth_d
            filter_3d = np.ones((smooth_T, d, d)) / (smooth_T * d * d)
            anomalymaps = convolve(anomalymaps, filter_3d)

    frame_scores = np.max(anomalymaps, axis=(1, 2))
    return frame_scores, anomalymaps
