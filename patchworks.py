#!/usr/bin/python

import itertools
import matplotlib
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import sklearn.decomposition
import scipy.spatial
import numpy

from patches import Patches
from utilities import crop, plot_3d_scatter


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
        self.__alternatives = alternatives
        self.__colorspace = colorspace
        self.__patch_shape = patch_shape
        real_shape = (patch_shape[0] * scale_factor,
                      patch_shape[1] * scale_factor)
        self.__images = [crop(image, real_shape) for image in images]
        # prepare images
        preprocessed = itertools.imap(self.preprocess, self.__images)
        data, mean, std, selected = self.__initialize_data(preprocessed, patch_shape)
        self.__mean = mean
        self.__std = std
        self.__selected = selected
        # prepare for comparisons
        max_components = int(0.5 * len(images))
        self.__pca = sklearn.decomposition.PCA(n_components=max_components)
        self.__pca.fit(data)
        self.__kd_tree = scipy.spatial.cKDTree(self.project(data))
        # save projected data for debugging
        self.__data = self.__pca.transform(data)

    @classmethod
    def __initialize_data(cls, images, patch_shape):
        # pull out into row vectors and stack
        rows = (image.flatten() for image in images)
        M = numpy.vstack(rows)
        # select columns with any variance
        std = M.std(0)
        selected = numpy.argwhere(std!=0)
        M = M[:, selected].squeeze()
        # get characteristics
        std = M.std(0)
        mean = M.mean(0)
        M = (M - mean) / std  # normalize
        return M, mean, std, selected

    # # # Helpers

    def normalize(self, data):
        """
        Normalize data.
        """
        return (data - self.__mean) / self.__std

    def prepare(self, patch):
        """
        Produce row vector ready for comparisions.
        """
        patch = self.preprocess(patch)
        data = self.select(patch)
        normalized = self.normalize(data)
        return self.project(normalized)

    def preprocess(self, patch):
        """
        Perform image processing on patch before flattened.
        """
        if patch.shape != self.__patch_shape:
            cropped = crop(patch, self.__patch_shape)
        if self.__colorspace == 'hsv':
            cropped = matplotlib.colors.rgb_to_hsv(cropped)
        return cropped

    def project(self, data):
        """
        Project the data onto the principal components.
        """
        return self.__pca.transform(data)

    def select(self, patch):
        """
        Produce a row vector from the patch.
        """
        return patch.flatten()[self.__selected].squeeze()

    # # # Main interface

    @property
    def component_images(self):
        """
        Returns images of the principal components of the library of images.
        """
        pca_images = (component.reshape(self.__patch_shape)
                      for component in self.__pca.components_)
        return pca_images

    def replace(self, patch):
        """
        Replace patch with one from library of images.
        """
        point = self.prepare(patch)
        alternatives = self.__alternatives
        dist, indexes = self.__kd_tree.query(point, k=alternatives)
        index = numpy.random.choice(indexes.flatten())
        return self.__images[index]

    def represent(self, image):
        """
        Create a patchwork representing the image.
        """
        patches = Patches(image, self.__patch_shape)
        replacement_patches = itertools.imap(self.replace, patches)
        return patches.stack(replacement_patches)

    def visualize(self, image):
        """
        Plots a visualization of the patchwork.

        The produced scatterplot shows a visualization of the library data
        (blue) and the patch data (red) in the projected space.
        """
        # data for patches
        patches = Patches(image, self.__patch_shape)
        patch_data = numpy.vstack(itertools.imap(self.prepare, patches))
        patch_data = patch_data[:, :3]  # select the three principal components
        # plot
        figure = matplotlib.pyplot.figure()
        axis = figure.add_subplot(111, projection='3d')
        plot_3d_scatter(axis, self.__data, color='b')
        plot_3d_scatter(axis, patch_data, color='r')
        matplotlib.pyplot.show()
