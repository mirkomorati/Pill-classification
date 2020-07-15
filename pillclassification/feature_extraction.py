import numpy as np
from skimage import color, measure, filters, img_as_float
from skimage.transform import resize
from pillclassification.functions import superpixel, image_segmentation, color_kmeans, recreate_image

from collections import Counter


def feature_extraction(image):
    """
    Extracts the features from an image.

    The pipeline is:
        - Segment the image using a superpixel method and its Region Adjacency Graph.
        - Get the most likely ROI using an heuristic metric.
        - Extract the features from the ROI.

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    hu: array
        Hu moments of the (hopefully right) region.
    color: array
        The dominant color in the (hopefully right) region as RGB.
    """
    sp_labels = superpixel(image)
    rag, labels = image_segmentation(image, sp_labels)
    labels_rgb = color.label2rgb(labels, image, kind='avg', bg_label=0)

    regions = measure.regionprops(labels)
    
    likeliness = []

    for r in regions:
        if r.area < 300:
            likeliness.append(1)
            continue
        ellipse_area = np.pi * r.major_axis_length / 2 * r.minor_axis_length / 2
        likeliness.append(abs(1 - r.solidity) + abs(1 - r.filled_area / ellipse_area))

    if len(likeliness) == 0:
        raise ValueError('Likeliness is empty') 
    region = regions[np.argmin(likeliness)]

    hu = region.moments_hu
    hu = -1 * np.sign(hu) * np.log10(np.abs(hu))

    minr, minc, maxr, maxc = region.bbox
    cropped = labels_rgb[minr:maxr, minc:maxc]
    for i in range(3):
        cropped[:,:,i] *= region.image
        
    kmeans, labels_ = color_kmeans(cropped, n_colors=3)

    count = Counter(labels_)
    rgb_val = kmeans.cluster_centers_[count.most_common()[0][0]]

    return hu, rgb_val


def generate_image(img, bg):
    """
    Generate an image given a segmented pill and a background.

    Randomizes position and scale, adds a fake shadow.

    Paramenters
    -----------
    img: ndarray
    bg: Path

    Returns
    -------
    image: ndarray
    """
    assert img.shape[-1] == 4

    img = img_as_float(img)
    img = img[:1500, :]

    width = np.random.randint(200, 250)
    img = resize(img, (int(img.shape[0] * (width / img.shape[1])), width), anti_aliasing=True)
    
    width = 600
    bg = imread(bg)
    bg = img_as_float(bg)
    bg = resize(bg, (int(bg.shape[0] * (width / bg.shape[1])), width), anti_aliasing=True)

    x, y = np.random.randint(50,200, 2)
    
    bg_h, bg_w, bg_d = bg.shape
    img_h, img_w, img_d = img.shape

    shadow = img.copy()
    shadow[shadow < 1] = 0
    sigma = np.random.rand() * 10 + 10
    shadow = filters.gaussian(shadow, sigma=sigma)

    offset_x, offset_y = np.random.randint(0, 15, 2)

    template = np.zeros((bg_h, bg_w, img_d))
    template_shadow = np.zeros((bg_h, bg_w, img_d))
    template_shadow[y + offset_y : y + offset_y + img_h, x + offset_x : x + offset_x + img_w, :] = shadow
    plt.figure()
    plt.imshow(template_shadow)
    template[y : y + img_h, x : x + img_w, :] = img

    mask = np.stack([template[:,:,3] for _ in range(3)], axis = 2)
    inv_mask = 1. - mask
    # Changing brightness / contrast
    template[:,:,:3] += np.random.rand() * 0.3 - 0.1 * mask
    template[:,:,:3] *= np.random.rand() * 0.5 - 0.1 + 1
    return bg[:,:,:3] * inv_mask + template[:,:,:3] * mask - template_shadow[:,:,:3] * inv_mask
