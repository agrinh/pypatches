#!/usr/bin/python

import itertools
import sklearn.decomposition
import scipy.spatial
import numpy

from matcher import PCAMatcher, MeanColorMatcher
from patches import Patches
from utilities import crop, visual_compare


class Patchworks(object):
    """
    Produces patchworks.

    I.e. reproduces an image from a set of images using the represent method.
    """

    def __init__(self, images, patch_shape, scale_factor=1, alternatives=1,
                 colorspace='rgb'):
        if colorspace not in ('rgb', 'hsv'):
            raise ValueError('Only supported colorspaces are rgb and hsv')
        # store parameters
        self.__colorspace = colorspace
        self.__patch_shape = patch_shape
        real_shape = (patch_shape[0] * scale_factor,
                      patch_shape[1] * scale_factor)
        self.__images = [crop(image, real_shape) for image in images]
        # prepare images
        preprocessed = itertools.imap(self.preprocess, self.__images)
        data = numpy.vstack(preprocessed)
        self.match = MeanColorMatcher(data, alternatives)

    # # # Helpers

    def preprocess(self, patch):
        """
        Perform image processing on patch before flattened.
        """
        if patch.shape != self.__patch_shape:
            cropped = crop(patch, self.__patch_shape)
        if self.__colorspace == 'hsv':
            cropped = matplotlib.colors.rgb_to_hsv(cropped)
        return cropped.flatten().astype(numpy.float)

    # # # Main interface

    def replace(self, patch):
        """
        Replace patch with one from library of images.
        """
        point = self.preprocess(patch)
        return self.__images[self.match(point)]

    def represent(self, image):
        """
        Create a patchwork representing the image.
        """
        patches = Patches(image, self.__patch_shape)
        replacement_patches = itertools.imap(self.replace, patches)
        return patches.stack(replacement_patches)

    def visualize(self, image):
        patches = Patches(image, self.__patch_shape)
        extract = lambda patch: self.match.transform(self.preprocess(patch))
        patch_data = numpy.vstack(itertools.imap(extract, patches))
        patch_data = patch_data[:, :3]  # select the three principal components
        visual_compare(self.match.data, patch_data)

    @property
    def component_images(self):
        """
        Returns images of the principal components of the library of images.
        """
        pca_images = (component.reshape(self.__patch_shape)
                      for component in self.match.components)
        return pca_images
