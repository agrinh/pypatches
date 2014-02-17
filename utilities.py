#!/usr/bin/python

import scipy.misc


def crop(image, new_shape):
    """
    Returns a cropped version of numpy array image with shape specified.

    Will resize and crop the image such that the minimum part of the image is
    removed and the cropped image contains the middle of the original one.
    """
    shape = image.shape
    # get ratios
    ratio = float(shape[0]) / shape[1]
    new_ratio = float(new_shape[0]) / new_shape[1]
    # index of the dimension to be set to the corresponding one in new_shape,
    # the other one is scaled
    dim_compare = ratio > new_ratio
    scale_factor = float(new_shape[dim_compare]) / shape[dim_compare]
    shape = list(shape)  # enable editing
    # set new shape
    shape[not dim_compare] = int(scale_factor * shape[not dim_compare])
    shape[dim_compare] = new_shape[dim_compare]
    image = scipy.misc.imresize(image, shape)
    # add padding such that the middle of the image is retrieved
    vertical_padding = int((image.shape[0] - new_shape[0]) / 2)
    horizontal_padding = int((image.shape[1] - new_shape[1]) / 2)
    # calulate edges
    left = horizontal_padding
    top = vertical_padding
    right = horizontal_padding + new_shape[1]
    bottom = vertical_padding + new_shape[0]
    # crop image
    return image[top:bottom, left:right]


def plot_3d_scatter(axis, data, color='b'):
    """
    Simple helper for plotting 3d scatter data.

    Plots the first three columns of data.
    """
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    axis.scatter(xs, ys, zs, c=color)
