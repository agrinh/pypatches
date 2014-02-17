#!/usr/bin/python

import collections
import itertools
import numpy

from utilities import crop


class Patches(collections.Iterable, collections.Sized):
    """
    Represents an image and allows iteration over patches.

    Patches may be manipulated and then sent to the stack method to put them
    together into an image.
    """

    def __init__(self, image, patch_shape):
        # ammount of vertical / horizontal splits
        self.__vsplits = int(image.shape[0] / patch_shape[0])
        self.__hsplits = int(image.shape[1] / patch_shape[1])
        # crop to the optimal shape
        optimal_shape = (patch_shape[0] * self.__vsplits,
                         patch_shape[1] * self.__hsplits)
        image = crop(image, optimal_shape)
        self.__image = image
        self.__patch_shape = patch_shape

    def split(self):
        """
        Split into patches.
        """
        vsplitted = numpy.vsplit(self.__image, self.__vsplits)
        hsplitted = (numpy.hsplit(vsplit, self.__hsplits) for vsplit in vsplitted)
        return itertools.chain(*hsplitted)

    def stack(self, patches):
        """
        Puts together the patches to match the shape of the original image.

        Assumes patches are created from this Patches. Undoes what self.split
        does to an image.
        """
        # stack together horizontally
        vsplits = (numpy.hstack(itertools.islice(patches, self.__hsplits))
                   for _ in xrange(self.__vsplits))
        # stack vertically
        return numpy.vstack(vsplits)

    def __iter__(self):
        return self.split()

    def __len__(self):
        return self.__vsplits * self.__hsplits
