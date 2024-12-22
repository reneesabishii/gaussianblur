#Create Gaussian blur on images, load image through terminal
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image

#load image and create array
def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)

def rgb2ycbcr(im):
    """Convert RGB to YCbCr."""
    xform = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]], dtype=np.float32)
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128.
    return ycbcr.astype(np.uint8)

def ycbcr2rgb(im):
    """Convert YCbCr to RGB."""
    xform = np.array([[1., 0., 1.402], [1, -0.34414, -0.71414], [1., 1.772, 0.]], dtype=np.float32)
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128.
    rgb = rgb.dot(xform.T)
    return np.clip(rgb, 0., 255.).astype(np.uint8)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')

    # Create a size by size ndarray of type float32

    # Populate the values of the kernel. Note that the middle `pixel` should be x = 0 and y = 0.

    # Normalize the values such that the sum of the kernel = 1

    #rv array with float32, values of the kernel
    #divide by sum of the values to normalize
    #kernelPix location of each pixel in kernel
    #rv calculate each value of the kernel 5x5 with the gaussian formula
    else:
        rv = np.empty([size, size], dtype=np.float32)
        kernelPix = int((size-1)/2)
        for x in range(-kernelPix, kernelPix+1):
            for y in range(-kernelPix, kernelPix +1):
                rv[x + kernelPix][y + kernelPix] = (1 / (2 * np.pi * sigma * sigma)) * np.exp(-(x * x + y * y) / (2 * sigma * sigma ))
        rv = np.divide(rv, np.sum(rv))
        return rv


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))

    # TODO: determine, using the kernel shape, the ith and jth locations to start at.

    # Check if the kernel stretches beyond the border of the image.
    #if .....:
        # if so, return the input pixel at that location.
    #else:
        # perform the convolution.
    #Get the values of the kernel pixel, k
    k = int((kernel.shape[0] -1) / 2)
    #Checking if the kernel stretches beyond the border of the image
    if i - k < 0 or j - k < 0 or i + k >= img.shape[0] or j + k >= img.shape[1]:
        return img[i][j]
    #perform convolution
    #return the output
    else:
        outputPix = []
        for m in range(-k, k+1):
            for n in range(-k, k+1):
                convolution = kernel[m + k][n + k] * img[i - m][j - n]
                outputPix.append(convolution)
        output = np.sum(np.array(outputPix))
        return output


def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    # Make a copy of the input image to save results

    #Populate each pixel in the input by calling convolve_pixel and return results.
    #Copy of input image
    #Array is dtype('unit8')
    #Call convolve pixel to return results
    results = np.empty(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = convolve_pixel(img, kernel, i, j)
            results[i][j] = pixel
    results = np.array(np.around(results), dtype=np.uint8)
    return results

def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    
    #split array into 3 channels
    #using np.squeeze I had to look up, because just splitting was giving two channels?
    channel = np.dsplit(img, 3)
    for i in range(0,3):
        channel[i] = np.squeeze(channel[i])
    (r, g, b) = channel

    return (r, g, b)


def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    # TODO: Implement me
    #Merge the three channels using dstack for all three arrays
    merge = np.dstack((r, g, b))
    return merge


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--ycbcr', action='store_true', help='Filter in YCbCr space')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')
    parser.add_argument('--subsample', type=int, default=1, help='Subsample by factor')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    if args.ycbcr:
        # Convert to YCbCr
        inputImage = rgb2ycbcr(inputImage)

        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (y, cb, cr) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                     (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with cb and cr
        logging.info('Convolving the Cb channel')
        cb = convolve(cb, kernel)
        logging.info('Convolving the Cr channel')
        cr = convolve(cr, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(y, cb, cr)

        # convert to RGB
        resultImage = ycbcr2rgb(resultImage)
    else:
        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (r, g, b) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                     (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with each input channel
        logging.info('Convolving the first channel')
        r = convolve(r, kernel)
        logging.info('Convolving the second channel')
        g = convolve(g, kernel)
        logging.info('Convolving the third channel')
        b = convolve(b, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(r, g, b)

    # subsample image
    if args.subsample != 1:
        # subsample by a factor of 2
        resultImage = resultImage[::args.subsample, ::args.subsample, :]

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
