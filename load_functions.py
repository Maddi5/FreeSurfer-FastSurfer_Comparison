"""

Script with the functions to load the volumes and extract the largest connected component

Author: Maddalena Cavallo
Date: July 2024

"""

import numpy as np
import nibabel as nib
from scipy.ndimage import label as connected_components



def load_volume(filename): 

    """
    This function loads and prepares the input volume for further processing.

    The function gets the data from the .mgz volumes and transpose them to adapt to the standard visualization.
    Every non-zero label from the FreeSurfer/FastSurfer segmentation is then set equal to True.


    Parameters
    ----------
    filename : str
        The path to the .mgz file to be loaded.

    Returns
    -------
    image_binary : numpy.ndarray
        A binary volume.

    Raises
    ------
    FileNotFoundError
        If the file specified by 'filename' does not exist.
    ValueError
        If the input image contains only null values.

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





def get_largest_CC(binary_volume):

    """
    Extract the largest connected component from a binary image.

    This function identifies and extracts the largest connected component (assumed to be the brain)
    from a given binary image. The input binary image is supposed to be the output from load_volume() function.

    Parameters
    ----------
    binary_volume : numpy.ndarray
        A 3D binary image where the brain and other structures are labelled as 1, and the background is labelled as 0.

    Returns
    -------
    largest_conn_component : numpy.ndarray
        A binary image of the same shape as `binary_volume` containing only the largest connected component (the brain).

    Raises
    ------
    ValueError
        If the input image is not a binary image.
    ValueError
        If no connected components are found in the input image.

    Notes
    -----
    The function uses a cube connectivity structure of shape (3, 3, 3) to label connected components.
    
    """

    #check if the input image is binary
    if not np.all((binary_volume == 0) | (binary_volume == 1)):
        raise ValueError("Input image is not binary")
    
    # to use cube connectivity
    cube = np.ones(shape=(3, 3, 3))

    #give a label to every connected component
    conn_components, _ = connected_components(binary_volume, structure=cube)

    #check if there aren't connected components
    if conn_components.max() == 0:
        raise ValueError("No connected components found in the input image")
    
    #get the largest component, excluding the background label
    largest_label = np.argmax(np.bincount(conn_components.flat)[1:]) + 1
    largest_conn_component = (conn_components == largest_label)

    return largest_conn_component