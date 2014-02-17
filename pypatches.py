#!/usr/bin/python

import glob
import optparse
import scipy.misc

import patchworks

def get_optionparser():
    usage = 'usage: %prog [options] library target output'
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-s', '--shape', default='10x10x3',
                      help='Shape of patches files, NxNxN')
    parser.add_option('-x', '--x_scale', default='1',
                      help='Multiply size of patches with this')
    parser.add_option('-a', '--alternatives', default='1',
                      help='Find more alternatives to patches')
    parser.add_option('-c', '--colorspace', default='rgb',
                      help='Colorspace, rgb or hsv')
    parser.add_option('--visualize', action='store_true', default=False,
                      help='Debug: produce plot [b == library, r == patches]')
    parser.add_option('--components', action='store_true', default=False,
                      help='Debug: produce images of the principal components')
    return parser


def main():
    parser = get_optionparser()
    options, arguments = parser.parse_args()
    if len(arguments) != 3:
        print 'Incorrect ammount of arguments passed'
        return
    # load images
    library, target, output = arguments
    filenames = [g for g in glob.glob('%s/*.jpg' % library)]
    images = map(scipy.misc.imread, filenames)
    shape = map(int, options.shape.split('x'))
    scale = int(options.x_scale)
    alt = int(options.alternatives)
    # create patchworks
    p = patchworks.Patchworks(images, shape, scale_factor=scale,
                              alternatives=alt, colorspace=options.colorspace)
    a = scipy.misc.imread(target)
    # represent a with a patchwork
    b = p.represent(a)
    scipy.misc.imsave(output, b)
    # perform extra actions selected
    if options.components:
        for i, component_image in enumerate(p.component_images):
            scipy.misc.imsave('%d_c_%s' % (i, output), component_image)
    if options.visualize:
        p.visualize(a)


if __name__ == '__main__':
    main()
