#Script with functions to compute the metrics

import numpy as np 
from skimage import metrics




def jaccard_index(seg1, seg2):

    """
    Compute the Jaccard Index between two binary volumes.

    The Jaccard Index is a similarity index, calculated as the intersection over the union of the two sets.
    It ranges between 0 and 1, with 0 indicating no overlap and 1 complete overlap.

    Parameters
    ----------
    seg1 : numpy.ndarray
        The first binary volume in the comparison.
    seg2 : numpy.ndarray
        The second binary volume in the comparison.

    Returns
    -------
    jacc_index : float
        The Jaccard Index between the two input volumes.

    Raises
    ------
    ValueError
        If the volumes to compare have different shapes.
    ValueError
        If both the volumes have only values equal to 0.
    ValueError
        If one of the volumes has only values equal to 0.
    ValueError
        If the volumes contain values other than 0 and 1 (not binary).

    Notes
    ------
    The intersection is computed as the product of the two binary images i.e. it is 1 only if both are 1.
    The union is computed as the total number of ones in both the images minus the intersection.

    Examples
    --------
    >>> seg1 = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    >>> seg2 = np.array([[[1, 1], [0, 0]], [[0, 1], [1, 0]]])
    >>> jaccard_index(seg1, seg2)
    0.3333333333333333
    """

    if seg1.shape != seg2.shape:
        raise ValueError("Input volumes are of different shape")

    if np.sum(seg1) == 0 and np.sum(seg2) == 0:
        raise ValueError("Both of the segmentations are empty. Jaccard Index cannot be computed")
        
    if np.sum(seg1) == 0 or np.sum(seg2) == 0:
        raise ValueError("One of the segmentations is empty. Jaccard Index cannot be computed")
        
    if not np.all(np.isin(seg1, [0, 1])) or not np.all(np.isin(seg2, [0,1])):
        raise ValueError("Input volumes contain values other than 0 and 1")


    intersection = np.sum(seg1 * seg2)

    union = np.sum(seg1) + np.sum(seg2) - intersection

    #Jaccard Index is computed as intersection over union of the two sets
    jacc_index = intersection / union
    
    return jacc_index




def volumetric_difference(seg1, seg2):

    """
    Compute the volumetric difference between two binary volumes.

    The volumetric difference is the difference between the volumes in the input images. The volumes are computed as the total number of voxels equal to 1.

    Parameters
    ----------
    seg1 : numpy.ndarray
        The first binary volume in the comparison.
    seg2 : numpy.ndarray
        The second binary volume in the comparison.

    Returns
    -------
    vol_difference : int
        The difference in volume (total number of voxels equal to 1) between the two input images.

    Raises
    ------
    ValueError
        If the volumes have different shapes.
    ValueError
        If the volumes contain values other than 0 and 1 (not binary).
    ValueError
        If both the volumes have only values equal to 0.
    ValueError
        If one of the volumes has only values equal to 0.

    Examples
    --------
    >>> seg1 = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 1]]])
    >>> seg2 = np.array([[[1, 0], [1, 0]], [[0, 1], [1, 0]]])
    >>> volumetric_difference(seg1, seg2)
    0
    """


    if seg1.shape != seg2.shape:
        raise ValueError("Input volumes are of different shape")

    if not np.all(np.isin(seg1, [0, 1])) or not np.all(np.isin(seg2, [0,1])):
        raise ValueError("Input volumes contain values other than 0 and 1. Check input data")
    
    volume_1 = np.sum(seg1)
    volume_2 = np.sum(seg2)

    if volume_1 == 0  and volume_2 == 0:
        raise ValueError("Both of the segmented volumes are empty. Check input data")

    if volume_1 == 0  or volume_2 == 0:
        raise ValueError("One of the segmented volumes is empty. Check input data")
    
    vol_difference = volume_1 - volume_2
    
    return vol_difference




def Hausdorff_distance(seg1,seg2):

    """
    Compute modified Hausdorff distance between non-zero elements of two binary volumes.

    This function computes the modified Hausdorff distance between two sets [1]_.
    The Hausdorff distance measures the extent to which each point of a segmentation set lies near some point of the other set

    Parameters
    ----------
    seg1 : numpy.ndarray
        The first binary volume in the comparison.
    seg2 : numpy.ndarray
        The second binary volume in the comparison. Must have the same shape of the first.

    Returns
    -------
    hausdorff_dist : float
        The modified Hausdorff distance between coordinates of nonzero pixels in the two input volumes.

    Raises
    ------
    ValueError
        If the input volumes have different shapes.
    ValueError
        If the volumes contain values other than 0 and 1 (not binary).
    ValueError
        If both the volumes have only values equal to 0.
    ValueError
        If one of the volumes has only values equal to 0.

    Notes
    -----
    The Modified Hausdorff Distance has been shown to perform better than the directed Hausdorff Distance in the following work by Dubuisson et al. [2]_

    References
    ----------
    .. [1] https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.hausdorff_distance 
    .. [2] M. P. Dubuisson and A. K. Jain. "A Modified Hausdorff distance for object matching". In ICPR94, pages A:566-568, Israel, 1994.

    """

    if seg1.shape != seg2.shape:
        raise ValueError("Input data are of different shape. Hausdorff distance cannot be computed")

    if not np.all(np.isin(seg1, [0, 1])) or not np.all(np.isin(seg2, [0,1])):
        raise ValueError("Input volumes contain values other than 0 and 1. Check input data")
    
    seg1 = np.asarray(seg1, dtype=bool)
    seg2 = np.asarray(seg2, dtype=bool)

    if np.sum(seg1) == 0 and np.sum(seg2) == 0:
        raise ValueError("Both of the segmentations are empty. Hausdorff distance cannot be computed")

    if np.sum(seg1) == 0 or np.sum(seg2) == 0:
        raise ValueError("One of the segmentations is empty. Hausdorff distance cannot be computed")
    

    hausdorff_dist = metrics.hausdorff_distance(seg1, seg2, method = 'modified')

    return(hausdorff_dist)