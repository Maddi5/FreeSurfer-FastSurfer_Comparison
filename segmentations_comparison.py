#!/usr/bin/env python3

import numpy as np
import nibabel as nib

from scipy.ndimage import label as connected_components
from medpy.metric.binary import dc


from setup import Setup_Script
from metrics import jaccard_index, volumetric_difference



Setup_Script()



def load_volume(filename): 
    """
    Aim: load and prepare MRI volume for further processing
    """
 
    try:
        image = nib.load(filename)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")


    # get the data
    image_array = image.get_fdata()

    #check input data are not empty
    if np.all(image_array == 0):
        raise ValueError("The input image contains only null values")

    #adapt to standard visualization
    image_array = image_array.transpose(1, 2, 0)[:, ::-1, :]
    
    image_binary = image_array != 0 # every label becomes True

    return image_binary





def get_largest_CC(segmentation):
    """
    Aim: extract a binary image of the largest connected component (the brain)
    """
    
    #check if the input image is segmented
    if not np.all((segmentation == 0) | (segmentation == 1)):
        raise ValueError("Input image is not segmented")
    
    # to use cube connectivity
    cube = np.ones(shape=(3, 3, 3))

    #give a label to every connected component
    conn_components, _ = connected_components(segmentation, structure=cube)

    #check if there aren't connected components
    if conn_components.max() == 0:
        raise ValueError("No connected components found in the input image")
    
    #get the largest component, exluding the background label
    largest_label = np.argmax(np.bincount(conn_components.flat)[1:]) + 1
    largest_conn_component = (conn_components == largest_label)

    return largest_conn_component




# Loading volumes
input_volumes= {
    'FreeSurfer': load_volume('Simulated_Data/Simulated_FreeSurfer.mgz'),
    'FreeSurfer_auto': load_volume('Simulated_Data/Simulated_FreeSurfer_auto.mgz'),
    'FastSurfer': load_volume('Simulated_Data/Simulated_FastSurfer.mgz')
}


data = {
    'FreeSurfer': get_largest_CC(input_volumes['FreeSurfer']).astype(int),
    'FreeSurfer_auto': get_largest_CC(input_volumes['FreeSurfer_auto']).astype(int),
    'FastSurfer': get_largest_CC(input_volumes['FastSurfer']).astype(int)
}





#Check if it works in the first case of comparison
dice_coeff_A = dc(data['FreeSurfer'], data['FreeSurfer_auto'])
jaccard_index_A = jaccard_index(data['FreeSurfer'], data['FreeSurfer_auto'])
vol_difference_A = volumetric_difference(data['FreeSurfer'], data['FreeSurfer_auto'])


print("Dice Similarity Coefficient:", dice_coeff_A)
print("Jaccard index:", jaccard_index_A)
print("Volume difference between the segmentations:", vol_difference_A)