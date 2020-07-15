from skimage import exposure, segmentation, filters, color, util, img_as_float
from skimage.transform import resize
from skimage.io import imread
from skimage.future import graph
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def image_equalization(image, method='percentile'):
    """
    Equalizes an image based on the method.
    
    Parameters
    ----------
    image: ndarray
    method: string, one of {'percentile', 'adapthist'}

    Returns
    -------
    image: ndarray
        Equalized image based on the chosen method.
    """
    if method == 'percentile':
        p1, p2 = np.percentile(image, (2,98))
        return exposure.rescale_intensity(image, in_range=(p1, p2))
    elif method == 'adapthist':
        return exposure.exualize_adapthist(image, clip_limit=0.01)
    return image


def image_sharpening(image, blur_size=10, unsharp_strength=1):
    blurred = filters.gaussian(image, blur_size)
    highpass = image - unsharp_strength * blurred
    return image + highpass


def superpixel(image, method='watershed'):
    """
    Segmentation based on superpixels.

    Parameters
    ----------
    image: ndarray
    method: string, one of {'slic', 'watershed'}

    Returns
    -------
    labels: ndarray
    """
    if method == 'slic':
        image = util.img_as_float(image)
        return segmentation.slic(image, n_segments=300, compactness=30, sigma=5, start_label=1)
    elif method == 'watershed':
        gradient = filters.sobel(color.rgb2gray(image))
        return segmentation.watershed(gradient, markers=150, compactness=0.0001)


def recreate_image(codebook, labels, w, h):
    """
    Recreate an image given a codebook and labels.
    
    Parameters
    ----------
    codebook: ndarray
        Has shape (n_clusters, n_features).
    labels: ndarray
        Index of the cluster each sample belongs to. 
    w, h: int

    Returns
    -------
    image: ndarray
    """
    d = codebook.shape[1]
    image = np.zeros((w,h,d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def color_kmeans(image, n_colors=3):
    """
    Clusters an image using kmeans.

    Parameters
    ----------
    image: ndarray
    n_colors: int, optional
        Number of clusters.

    Returns
    -------
    kmeans: KMeans instance
    labels: ndarray
        Index of the cluster each sample belongs to.
    """
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
    """
    Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def _weight_mean_color(graph, src, dst, n):
    """
    Callback to handle merging nodes by recomputing mean color.

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
    """
    Callback called before merging two nodes of a mean color distance graph.

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


def merge_hier_boundary(labels, image, thresh=0.03, show_rag=False):
    """
    Merges the given labels using a RAG based on boundaries.

    Parameters
    ----------
    labels: ndarray
    image: ndarray
    thresh: float
    show_rag: bool
    
    Returns
    -------
    rag: RAG
    labels: ndarray
        Merged labels.
    """
    edges = filters.sobel(color.rgb2gray(image))
    rag = graph.rag_boundary(labels, edges)
    rag_copy = False
    if show_rag:
        rag_copy = True
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    labels = graph.merge_hierarchical(labels, rag, thresh=thresh, rag_copy=rag_copy,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)
    if show_rag:
        graph.show_rag(labels, rag, image, ax=ax[0])
        ax[0].title('Initial RAG')
        graph.show_rag(labels, graph.rag_boundary(labels, edges), ax=ax[1])
        ax[1].title('Final RAG')

    return rag, labels


def merge_hier_color(labels, image, thresh=0.08, show_rag=False):
    """
    Merges the given labels using a RAG based on the mean color.

    Parameters
    ----------
    labels: ndarray
    image: ndarray
    thresh: float
    show_rag: bool

    Returns
    -------
    rag: RAG
    labels: ndarray
        Merged labels.
    """
    rag = graph.rag_mean_color(image, labels)
    rag_copy = False
    if show_rag:
        rag_copy = True
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    labels = graph.merge_hierarchical(labels, rag, thresh=thresh, rag_copy=rag_copy,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
    # labels2 = graph.cut_normalized(img_slic, rag, thresh=30)
    # labels2 = graph.cut_threshold(img_slic, rag, 0.2)
    if show_rag:
        graph.show_rag(labels, rag, image, ax=ax[0])
        ax[0].title('Initial RAG')
        graph.show_rag(labels, graph.rag_mean_color(image, labels), ax=ax[1])
        ax[1].title('Final RAG')

    return rag, labels


def image_segmentation(image, labels, method='color', thresh=0.08, show_rag=False):
    """
    Segments an image given its labels using a RAG. 

    Parameters
    ----------
    labels: ndarray
    image: ndarray
    method: string, one of {'color', 'boundary'}
    thresh: float
    show_rag: bool
    
    Returns
    -------
    rag: RAG
    labels: ndarray
    """
    if method == 'boundary':
        return merge_hier_boundary(labels, image)
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
    template[:,:,:3] *= (np.random.rand() * 0.5 - 0.2 + 1) * mask
    return bg[:,:,:3] * inv_mask + template[:,:,:3] * mask - template_shadow[:,:,:3] * inv_mask
