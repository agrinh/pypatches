#!/usr/bin/python
"""
pypatches.
"""
import numpy
import scipy.spatial
import sklearn.decomposition

class Patches(object):

    def __init__(self, images, patch_shape, scale_factor=1, alternatives=1):
        self.__patch_shape = patch_shape
        self.__alternatives = alternatives
        real_shape = (patch_shape[0] * scale_factor, patch_shape[1] * scale_factor)
        self.__images = [ self.crop(image, real_shape) for image in images ]
        max_components = int(0.1 * numpy.prod(patch_shape))
        self.__pca = sklearn.decomposition.PCA(n_components=max_components)
        images = [ self.crop(image, patch_shape) for image in images ]
        data, mean, std = self.__initialize_data(images, patch_shape)
        self.__mean = mean
        self.__std = std
        self.__pca.fit(data)
        self.__kd_tree = scipy.spatial.cKDTree(self.project(data))

    @classmethod
    def __initialize_data(cls, images, patch_shape):
        size = numpy.prod(patch_shape)
        reshaped = (image.reshape((1, size)) for image in images)
        M = numpy.vstack(reshaped)
        mean = M.mean(0)
        std = M.std(0)
        M = (M - mean) / std  # normalize
        return M, mean, std

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
        dim_compare = int(ratio > new_ratio)
        scale_factor = float(new_shape[dim_compare]) / shape[dim_compare]
        shape = list(shape)
        shape[0:2] = (int(scale_factor * shape[index]) for index in range(2))
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
        normalized = self.normalize(patch.reshape((1, patch.size)))
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
        parser.add_option('-s', '--shape', default='10x10x3', help='Shape of patches files, NxN')
        parser.add_option('-x', '--x_scale', default='1', help='Multiply size of patches with this')
        parser.add_option('-a', '--alternatives', default='1', help='Find more alternatives to patches')
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
        p = Patches(images, shape, scale_factor=scale, alternatives=alt)
        a = scipy.misc.imread(target)
        b = p.represent(a)
        scipy.misc.imsave(output, b)
