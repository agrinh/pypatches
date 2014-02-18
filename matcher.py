#!/usr/bin/python

import abc
import scipy.spatial
import sklearn.decomposition


class Matcher(object):
    """
    Base class for Matchers.

    A Matcher should take a library of numpy array images in an iterable and
    find the the image in the library best matching ones passed for
    comparision.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, image_library, randomization=1):
        """
        Recieve image library to compare images to.

        Randomization sets how many close matches to choose from.
        """

    def __call__(self, image):
        """
        Returns the result from the match method.
        """
        return self.match(point)

    @abc.abstractmethod
    def match(self, image):
        """
        Returns the index of the image in images closes matching the image.
        """


class MeanColorMatcher(Matcher):
    """
    Compare by average of colors.
    """

    def __init__(self, image_library, randomization=1):
        self.__image_library = list(image_library)
        self.__randomization = randomization
        library_averages = itertools.imap(self.color_average, image_library)
        library_averages = numpy.array(library_averages)  # convert to numpy
        self.__kd_tree = scipy.spatial.cKDTree(library_averages)

    @staticmethod
    def color_average(image):
        flat = image.flatten()  # will be [R, G, B, R, G, B, ...]
        averages = (flat[0::3].mean(), flat[1::3].mean(), flat[2::3].mean())
        return numpy.array(averages)

    def match(self, image):
        averages = self.color_average(image)
        alternatives = self.__alternatives
        dist, indexes = self.__kd_tree.query(averages, k=self.__randomization)
        index = numpy.random.choice(indexes.flatten())
        return self.__image_library[index]

class NormalizingMatcher(Matcher):
    pass


class PCAMatcher(NormalizingMatcher):
