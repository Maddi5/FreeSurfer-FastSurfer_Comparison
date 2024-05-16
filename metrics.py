#Script with functions to compute the metrics

import numpy as np 
from skimage import metrics




def jaccard_index(seg1, seg2):


    if seg1.shape != seg2.shape:
        raise ValueError("Segmentations are of different shape")

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

    if seg1.shape != seg2.shape:
        raise ValueError("Input segmentations are of different shape")

    if not np.all(np.isin(seg1, [0, 1])) or not np.all(np.isin(seg2, [0,1])):
        raise ValueError("Segmentations contain values other than 0 and 1. Check input data")
    
    volume_1 = np.sum(seg1)
    volume_2 = np.sum(seg2)

    if volume_1 == 0  and volume_2 == 0:
        raise ValueError("Both of the segmented volumes are empty. Check input data")

    if volume_1 == 0  or volume_2 == 0:
        raise ValueError("One of the segmented volumes is empty. Check input data")
    
    vol_difference = volume_1 - volume_2
    
    return vol_difference




def Hausdorff_distance(seg1,seg2):

    seg1 = np.asarray(seg1, dtype=bool)
    seg2 = np.asarray(seg2, dtype=bool)

    hausdorff_dist = metrics.hausdorff_distance(seg1, seg2, method = 'modified')

    return(hausdorff_dist)