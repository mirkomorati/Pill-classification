from skimage.measure import moments_normalized, moments_central, moments_hu
from numpy import sign, log10, abs, sum, divide, ndarray

def match_shapes(img_a: ndarray, img_b: ndarray):
    ''' 
        This function takes in input two images and returns
        the distances between the images using the HU moments.
    '''

    # calculating the hu moments
    hu_a = moments_hu(moments_normalized(moments_central(img_a)))
    hu_b = moments_hu(moments_normalized(moments_central(img_b)))

    # changing to log scale 
    hu_a = -1 * sign(hu_a) * log10(abs(hu_a))
    hu_b = -1 * sign(hu_b) * log10(abs(hu_b))

    # calculating 3 distaces 
    d1 = sum(abs((1/hu_b) - (1/hu_a)))
    d2 = sum(abs(hu_b - hu_a))
    d3 = sum(divide(abs(hu_a - hu_b), abs(hu_a)))

    # returning the distances
    return d1, d2, d3 
