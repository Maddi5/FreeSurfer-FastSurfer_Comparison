#Script with functions to compute the metrics

import numpy as np 



def jaccard_index(seg_1, seg_2):


    if seg_1.shape != seg_2.shape:
        raise ValueError("Segmentations are of different shape")

    if np.sum(seg_1) == 0 and np.sum(seg_2) == 0:
        raise ValueError("Both of the segmentations are empty. Jaccard Index cannot be computed")
        
    if np.sum(seg_1) == 0 or np.sum(seg_2) == 0:
        raise ValueError("One of the segmentations is empty. Jaccard Index cannot be computed")
        
    if not np.all(np.isin(seg_1, [0, 1])) or not np.all(np.isin(seg_2, [0,1])):
        raise ValueError("Input volumes contain values other than 0 and 1")


    intersection = np.sum(seg_1 * seg_2)

    union = np.sum(seg_1) + np.sum(seg_2) - intersection

    #Jaccard Index is computed as intersection over union of the two sets
    jacc_index = intersection / union
    
    return jacc_index



def volumetric_difference(seg_1, seg_2):

    volume_1 = np.sum(seg_1)
    volume_2 = np.sum(seg_2)

    vol_difference = volume_1 - volume_2
    
    return vol_difference



