import cv2
import numpy as np

def generate_sliding_windows(im_shape, ws=(800, 800), s=None):
    im_h, im_w = im_shape[:2]

    window_h, window_w = np.array(ws).astype(int)[:2]
    s = ws if s is None else s
    stride_h, stride_w = np.array(s).astype(int)[:2]

    ys = np.arange(0, im_h - window_h + 1, stride_h)
    xs = np.arange(0, im_w - window_w + 1, stride_w)

    thresh_h = int(im_h * 0.001)
    thresh_w = int(im_w * 0.001)

    if xs[-1] + window_w < im_w - thresh_w:
        xs = np.append(xs, im_w - window_w)

    if ys[-1] + window_h < im_h - thresh_h:
        ys = np.append(ys, im_h - window_h)

    xx, yy = np.meshgrid(xs, ys)
    x1 = xx.ravel()
    y1 = yy.ravel()
    x2 = x1 + window_w
    y2 = y1 + window_h

    windows = np.stack([x1, y1, x2, y2], axis=1)

    return windows

def sliding_crop(image, ws=(800, 800), s=None):
    coords = generate_sliding_windows(image.shape, ws=ws, s=s)

    crops = [
        image[y1:y2, x1:x2]
        for (x1, y1, x2, y2) in coords
    ]
    return crops, coords