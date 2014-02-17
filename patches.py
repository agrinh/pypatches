#!/usr/bin/python
"""
pypatches.
"""
import itertools
import matplotlib
import numpy
import scipy.spatial
import sklearn.decomposition


class Patches(object):

    def __init__(self, images, patch_shape, scale_factor=1, alternatives=1,
                 colorspace='rgb'):
        if colorspace not in ('rgb', 'hsv'):
            raise ValueError('Only supported colorspaces are rgb and hsv')
        # store parameters
        self.__patch_shape = patch_shape
        self.__alternatives = alternatives
        self.__colorspace = colorspace
        real_shape = (patch_shape[0] * scale_factor,
                      patch_shape[1] * scale_factor)
        self.__images = [self.crop(image, real_shape) for image in images]
        # prepare images
        preprocessed = itertools.imap(self.preprocess, self.__images)
        data, mean, std, selected = self.__initialize_data(preprocessed, patch_shape)
        self.__mean = mean
        self.__std = std
        self.__selected = selected
        # prepare for comparisons
        max_components = int(0.1 * numpy.prod(patch_shape))
        self.__pca = sklearn.decomposition.PCA(n_components=max_components)
        self.__pca.fit(data)
        self.__kd_tree = scipy.spatial.cKDTree(self.project(data))

    @classmethod
    def __initialize_data(cls, images, patch_shape):
        # pull out into a row vector
        size = numpy.prod(patch_shape)
        reshaped = (image.reshape((1, size)) for image in images)
        # stack rows
        M = numpy.vstack(reshaped)
        # select those with any variance
        std = M.std(0)
        selected = numpy.argwhere(std!=0)
        M = M[:, selected].squeeze()
        # get characteristics
        std = M.std(0)
        mean = M.mean(0)
        M = (M - mean) / std  # normalize
        return M, mean, std, selected

    def preprocess(self, image):
        cropped = self.crop(image, self.__patch_shape)
        if self.__colorspace == 'hsv':
            return matplotlib.colors.rgb_to_hsv(cropped)
        elif self.__colorspace == 'rgb':
            return cropped

    @staticmethod
    def crop(image, new_shape):
        """
        Returns a cropped version of image with shape specified.

        Will resize and crop the image such that the minimum part of the image is
        removed and the cropped image contains the middle of the original one.
        """
        shape = image.shape
        ratio = float(shape[0]) / shape[1]
        new_ratio = float(new_shape[0]) / new_shape[1]
        dim_compare = ratio > new_ratio
        scale_factor = float(new_shape[dim_compare]) / shape[dim_compare]
        shape = list(shape)  # enable editing
        shape[not dim_compare] = int(scale_factor * shape[not dim_compare])
        shape[dim_compare] = new_shape[dim_compare]
        image = scipy.misc.imresize(image, shape)
        vertical_padding = int((image.shape[0] - new_shape[0]) / 2)
        horizontal_padding = int((image.shape[1] - new_shape[1]) / 2)
        left = horizontal_padding
        top = vertical_padding
        right = horizontal_padding + new_shape[1]
        bottom = vertical_padding + new_shape[0]
        return image[top:bottom, left:right]

    def normalize(self, data):
        return (data - self.__mean) / self.__std

    def project(self, data):
        return self.__pca.transform(data)

    def replace(self, patch):
        # stretch into row vector and select
        patch = self.preprocess(patch)
        data = patch.reshape((1, patch.size))[:, self.__selected].squeeze()
        normalized = self.normalize(data)
        point = self.project(normalized)
        alternatives = self.__alternatives
        dist, indexes = self.__kd_tree.query(point, k=alternatives)
        index = numpy.random.choice(indexes.flatten())
        return self.__images[index]

    def represent(self, image):
        shape = list(image.shape)
        patch_shape = self.__patch_shape
        msplits = int(shape[0] / patch_shape[0])
        nsplits = int(shape[1] / patch_shape[1])
        shape = (patch_shape[0] * msplits, patch_shape[1] * nsplits)
        image = self.crop(image, shape)
        vsplits = numpy.vsplit(image, msplits)
        hsplits = (numpy.hsplit(vsplit, nsplits) for vsplit in vsplits)
        hsplits = (map(self.replace, hsplit) for hsplit in hsplits)
        vsplits = map(numpy.hstack, hsplits)
        new_image = numpy.vstack(vsplits)
        return new_image


if __name__ == '__main__':
    import glob
    import optparse
    import scipy.misc

    def get_optionparser():
        usage = 'usage: %prog [options] library target output'
        parser = optparse.OptionParser(usage=usage)
        parser.add_option('-s', '--shape', default='10x10x3', help='Shape of patches files, NxNxN')
        parser.add_option('-x', '--x_scale', default='1', help='Multiply size of patches with this')
        parser.add_option('-a', '--alternatives', default='1', help='Find more alternatives to patches')
        parser.add_option('-c', '--colorspace', default='rgb', help='Colorspace, rgb or hsv')
        return parser


    parser = get_optionparser()
    options, arguments = parser.parse_args()
    if len(arguments) != 3:
        print 'Incorrect ammount of arguments passed'
    else:
        library, target, output = arguments
        filenames = [g for g in glob.glob('%s/*.jpg' % library)]
        images = map(scipy.misc.imread, filenames)
        shape = map(int, options.shape.split('x'))
        scale = int(options.x_scale)
        alt = int(options.alternatives)
        p = Patches(images, shape, scale_factor=scale, alternatives=alt,
                    colorspace=options.colorspace)
        a = scipy.misc.imread(target)
        b = p.represent(a)
        scipy.misc.imsave(output, b)
