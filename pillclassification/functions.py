from skimage import exposure, segmentation, filters, color, util
from skimage.future import graph
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def image_equalization(image, method='percentile'):
    if method == 'percentile':
        p1, p2 = np.percentile(image, (2,98))
        return exposure.rescale_intensity(image, in_range=(p1, p2))
    return image

def image_sharpening(image, blur_size=10, unsharp_strength=1):
    blurred = filters.gaussian(image, blur_size)
    highpass = image - unsharp_strength * blurred
    return image + highpass

def superpixel(image, method='watershed'):
    if method == 'slic':
        image = util.img_as_float(image)
        return segmentation.slic(image, n_segments=300, compactness=30, sigma=5, start_label=1)
    elif method == 'watershed':
        gradient = filters.sobel(color.rgb2gray(image))
        return segmentation.watershed(gradient, markers=150, compactness=0.0001)

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w,h,d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def color_kmeans(image, n_colors=3):
    w, h, d = image.shape
    assert d == 3
    
    image_array = np.reshape(image, (w*h,d))

    samples = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(samples)

    return kmeans, kmeans.predict(image_array)

def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }

def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def merge_hier_boundary(labels, edges, image, thresh=0.03):
    rag = graph.rag_boundary(labels, edges)
    # graph.show_rag(labels, rag, image)
    # plt.title('Initial RAG')

    labels = graph.merge_hierarchical(labels, rag, thresh=thresh, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)
    return rag, labels

def merge_hier_color(labels, image, thresh=0.08):
    rag = graph.rag_mean_color(image, labels)
    # graph.show_rag(labels, rag, image)
    # plt.title('Initial RAG')
    labels = graph.merge_hierarchical(labels, rag, thresh=thresh, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
    # labels2 = graph.cut_normalized(img_slic, rag, thresh=30)
    # labels2 = graph.cut_threshold(img_slic, rag, 0.2)
    return rag, labels

def image_segmentation(image, labels, method='color'):
    if method == 'boundary':
        edges = filters.sobel(color.rgb2gray(image))
        return merge_hier_boundary(labels, edges, image)
    elif method == 'color':
        return merge_hier_color(labels, image)

def crop_center(image: np.ndarray, crop_scale: float) -> np.ndarray:
    ''' It returns the image cropped maintaing the same center and changing 
        the size of the crop_scale '''
    
    if crop_scale < 0 or crop_scale > 1.:
        raise ValueError('The crop scale must be less positive and less then 1')
    
    y,x,d = image.shape
    cropx = int(x * crop_scale)
    cropy = int(y * crop_scale)
    startx = int(x/2-(cropx/2))
    starty = int(y/2-(cropy/2)) - 100 

    return image[starty:starty+cropy,startx:startx+cropx,:]