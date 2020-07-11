import numpy as np
from skimage import color, measure
from pillclassification.functions import superpixel, image_segmentation, color_kmeans, recreate_image

from collections import Counter


def feature_extraction(image):
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
