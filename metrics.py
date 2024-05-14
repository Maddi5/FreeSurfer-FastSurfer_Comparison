#Script with functions to compute the metrics

import numpy as np 



def jaccard_index(seg_1, seg_2):

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



