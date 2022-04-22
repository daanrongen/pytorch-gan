import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scipy.spatial import distance
import random
import math
import glob


def center_crop(image, size):
    w, h = image.size
    left = (w - size) / 2
    top = (h - size) / 2
    right = (w + size) / 2
    bottom = (h + size) / 2
    center_crop = image.crop((left, top, right, bottom))
    return center_crop


def find_left(
    thresh1, thresh2, w_pad, h_pad, np_img, r_global_med, g_global_med, b_global_med
):
    left = 0
    for i in range(0, w_pad):
        r_stdev = np.std(np_img[h_pad:-h_pad, i : i + 1, 0:1])
        g_stdev = np.std(np_img[h_pad:-h_pad, i : i + 1, 1:2])
        b_stdev = np.std(np_img[h_pad:-h_pad, i : i + 1, 2:3])
        if r_stdev * r_stdev + g_stdev * g_stdev + b_stdev * b_stdev > thresh1:
            break

        r_med = np.median(np_img[h_pad:-h_pad, i : i + 1, 0:1])
        g_med = np.median(np_img[h_pad:-h_pad, i : i + 1, 1:2])
        b_med = np.median(np_img[h_pad:-h_pad, i : i + 1, 2:3])
        dst = distance.euclidean(
            (r_med, g_med, b_med), (r_global_med, g_global_med, b_global_med)
        )
        if dst < thresh2:
            break

        left = left + 1
    return left


def find_top(
    thresh1, thresh2, w_pad, h_pad, np_img, r_global_med, g_global_med, b_global_med
):
    top = 0
    for i in range(0, h_pad):
        r_stdev = np.std(np_img[i : i + 1, w_pad:-w_pad, 0:1])
        g_stdev = np.std(np_img[i : i + 1, w_pad:-w_pad, 1:2])
        b_stdev = np.std(np_img[i : i + 1, w_pad:-w_pad, 2:3])
        if r_stdev * r_stdev + g_stdev * g_stdev + b_stdev * b_stdev > thresh1:
            break

        r_med = np.median(np_img[i : i + 1, w_pad:-w_pad, 0:1])
        g_med = np.median(np_img[i : i + 1, w_pad:-w_pad, 1:2])
        b_med = np.median(np_img[i : i + 1, w_pad:-w_pad, 2:3])
        dst = distance.euclidean(
            (r_med, g_med, b_med), (r_global_med, g_global_med, b_global_med)
        )
        if dst < thresh2:
            break

        top = top + 1
    return top


def find_right(
    w, thresh1, thresh2, w_pad, h_pad, np_img, r_global_med, g_global_med, b_global_med
):
    right = w
    for i in range(0, w_pad):
        r_stdev = np.std(np_img[h_pad:-h_pad, w - i - 1 : w - i, 0:1])
        g_stdev = np.std(np_img[h_pad:-h_pad, w - i - 1 : w - i, 1:2])
        b_stdev = np.std(np_img[h_pad:-h_pad, w - i - 1 : w - i, 2:3])
        if r_stdev * r_stdev + g_stdev * g_stdev + b_stdev * b_stdev > thresh1:
            break

        r_med = np.median(np_img[h_pad:-h_pad, w - i - 1 : w - i, 0:1])
        g_med = np.median(np_img[h_pad:-h_pad, w - i - 1 : w - i, 1:2])
        b_med = np.median(np_img[h_pad:-h_pad, w - i - 1 : w - i, 2:3])
        dst = distance.euclidean(
            (r_med, g_med, b_med), (r_global_med, g_global_med, b_global_med)
        )
        if dst < thresh2:
            break

        right = right - 1
    return right


def find_bottom(
    h, thresh1, thresh2, w_pad, h_pad, np_img, r_global_med, g_global_med, b_global_med
):
    for i in range(0, h_pad):
        r_stdev = np.std(np_img[h - i - 1 : h - i, w_pad:-w_pad, 0:1])
        g_stdev = np.std(np_img[h - i - 1 : h - i, w_pad:-w_pad, 1:2])
        b_stdev = np.std(np_img[h - i - 1 : h - i, w_pad:-w_pad, 2:3])
        if r_stdev * r_stdev + g_stdev * g_stdev + b_stdev * b_stdev > thresh1:
            break

        r_med = np.median(np_img[h - i - 1 : h - i, w_pad:-w_pad, 0:1])
        g_med = np.median(np_img[h - i - 1 : h - i, w_pad:-w_pad, 1:2])
        b_med = np.median(np_img[h - i - 1 : h - i, w_pad:-w_pad, 2:3])
        dst = distance.euclidean(
            (r_med, g_med, b_med), (r_global_med, g_global_med, b_global_med)
        )
        if dst < thresh2:
            break

        h = h - 1
    return h


def remove_frames(image, thresh1=15000, thresh2=30):
    np_img = np.asarray(image)

    w = image.width
    h = image.height
    pad = 30
    w_pad = w // pad
    h_pad = h // pad

    r_global_med = np.median(np_img[h_pad:-h_pad, w_pad:-w_pad, 0:1])
    g_global_med = np.median(np_img[h_pad:-h_pad, w_pad:-w_pad, 1:2])
    b_global_med = np.median(np_img[h_pad:-h_pad, w_pad:-w_pad, 2:3])

    left = find_left(
        thresh1,
        thresh2,
        w_pad,
        h_pad,
        np_img,
        r_global_med,
        g_global_med,
        b_global_med,
    )

    top = find_top(
        thresh1,
        thresh2,
        w_pad,
        h_pad,
        np_img,
        r_global_med,
        g_global_med,
        b_global_med,
    )

    right = find_right(
        w,
        thresh1,
        thresh2,
        w_pad,
        h_pad,
        np_img,
        r_global_med,
        g_global_med,
        b_global_med,
    )

    bottom = find_bottom(
        h,
        thresh1,
        thresh2,
        w_pad,
        h_pad,
        np_img,
        r_global_med,
        g_global_med,
        b_global_med,
    )

    cropped_img = image.crop((left, top, right, bottom))
    return cropped_img
