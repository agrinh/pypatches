#!/usr/bin/python

import abc
import numpy
import scipy.spatial
import sklearn.decomposition
import sklearn.preprocessing


class Matcher(object):
    """
    Base class for Matchers.

    A Matcher should takes an MxN matrix data with M samples in N dimensions
    and find the closest matching sample to the ones passed for comparision.
    """
    __metaclass__ = abc.ABCMeta

    def __call__(self, image):
        """
        Returns the result from the match method.
        """
        return self.match(image)

    @abc.abstractmethod
    def match(self, image):
        """
        Returns the index of the image in images closes matching the image.
        """


class MeanColorMatcher(Matcher):
    """
    Compare by average of colors.
    """

    def __init__(self, data, randomization=1):
        self.__randomization = randomization
        color_averages = [self.color_average(row) for row in data]
        color_averages = numpy.array(color_averages)  # convert to numpy array
        self.__kd_tree = scipy.spatial.cKDTree(color_averages)

    @staticmethod
    def color_average(data):
        # data is expected to be a flattened image, it should be:
        # [R, G, B, R, G, B, ...]
        averages = (data[0::3].mean(), data[1::3].mean(), data[2::3].mean())
        return numpy.array(averages)

    def match(self, data):
        averages = self.color_average(data)
        dist, indexes = self.__kd_tree.query(averages, k=self.__randomization)
        if self.__randomization > 1:
            index = numpy.random.choice(indexes.flatten())
        else:
            index = indexes.flatten()[0]
        return index


class PCAMatcher(Matcher):

    MAX_COMPONENT_COEFFICIENT = 0.5

    def __init__(self, data, randomization=1):
        """
        Takes an MxN matrix data with M samples in N dimensions.
        """
        self.__randomization = randomization
        # pull out into row vectors, stack and scale
        self.__scaler = sklearn.preprocessing.StandardScaler()
        data = self.__scaler.fit_transform(data)
        # setup PCA and kd-tree
        max_components = int(self.MAX_COMPONENT_COEFFICIENT * len(data))
        self.__pca = sklearn.decomposition.PCA(n_components=max_components)
        data = self.__pca.fit_transform(data)
        self.__kd_tree = scipy.spatial.cKDTree(data)
        # store data
        self.__data = data

    def match(self, data):
        data = self.transform(data)
        dist, indexes = self.__kd_tree.query(data, k=self.__randomization)
        if self.__randomization > 1:
            index = numpy.random.choice(indexes.flatten())
        else:
            index = indexes
        return index

    def transform(self, data):
        data = self.__scaler.transform(data)
        data = self.__pca.transform(data)
        return data

    @property
    def components(self):
        return self.__pca.components_

    @property
    def data(self):
        return self.__data
