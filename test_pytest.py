#!/usr/bin/env python3

import numpy as np
import nibabel as nib
import pytest
from math import isclose, pi

from segmentations_comparison import load_volume, get_largest_CC
from metrics import jaccard_index, volumetric_difference




#LOAD_VOLUME

def test_load_volume_type_shape():
    """
    Aim: check type and shape of the output
    """
 
    assert isinstance(load_volume('Simulated_Data/Simulated_FreeSurfer.mgz'), np.ndarray), "Expected output type to be numpy.ndarray"

    assert load_volume('Simulated_Data/Simulated_FreeSurfer.mgz').shape == (256, 256, 256), "Expected loaded volume to have shape [256, 256, 256]"



def test_load_volume_segmented():
    """
    Aim: check that the output image is segmented
    """

    assert np.all((load_volume('Simulated_Data/Simulated_FreeSurfer.mgz') == 0) | (load_volume('Simulated_Data/Simulated_FreeSurfer.mgz') == 1)), "Expected output volume to be segmented"



def test_load_volume_transpose():
    """
    Aim: check if the values are correctly trasposed
    """

    test_volume = np.array([[[0, 1], [2, 4]],
                           [[3, 0], [1, 2]]])
    
    nib.save(nib.Nifti1Image(test_volume, np.eye(4)), 'test_volume.mgz') 

    loaded_volume = load_volume('test_volume.mgz')

    #transposed and binarized input image
    expected_output = np.array([[[1, 0], [0, 1]],
                            [[1, 1], [1, 1]]])

    assert np.array_equal(loaded_volume, expected_output), "Failed to correctly transpose the volume"



def test_load_volume_not_existent():
    """
    Aim: check the function to have the right behaviour when the filename doesn't exist
    """
    try:
        load_volume('non_existent_file.mgz')
    except FileNotFoundError as e:
        assert str(e) == "File non_existent_file.mgz not found", "Expected a FileNotFoundError when input image doesn't exist"



def test_load_volume_empty_image():
    """
    Aim: Check the function to raise an exception when input image is empty 
    """
 
    # create a test image with null values
    empty_image = np.zeros((2, 2, 2))
    nib.save(nib.Nifti1Image(empty_image, np.eye(4)), 'empty_image.mgz')

    try:
        load_volume('empty_image.mgz')
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError when the image contains only null values"





#GET_LARGEST_CC

def test_get_largest_CC_type_shape():
        """
        Aim: Check if the type and the shape of the output are the same as the input ones
        """
        input_image = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')

        output_image = get_largest_CC(input_image)

        assert isinstance(output_image, np.ndarray), "Expected output type to be numpy.ndarray"

        assert output_image.shape == input_image.shape, "Expected output shape to be the same as input"



def test_get_largest_CC_check_output():
    """
    Aim: Check if the function correctly identifies the largest connected component
    """
    # Create an input image with two connected components
    input_image = np.zeros((9, 9, 9))
    input_image[1:2, 1:2, 1:2] = 1  # Smaller component
    input_image[5:8, 5:8, 5:8] = 1  # Larger component

    output_image = get_largest_CC(input_image)

    expected_output = np.zeros((9, 9, 9))
    expected_output[5:8, 5:8, 5:8] = 1

    assert np.array_equal(output_image, expected_output), "Failed in identifying the largest connected component correctly"


def test_get_largest_CC_single_connected_component():
    """
    Aim: Check if the function correctly handles an input image with a single connected component
    """

    #Create image with a single connected component
    input_image = np.zeros((10, 10, 10))
    input_image[3:7, 3:7, 3:7] = 1

    output_image = get_largest_CC(input_image)

    assert np.array_equal(output_image, input_image), "Expected output to be the same as input for single connected component"



def test_get_largest_CC_full_input():
    """
    Aim: Check if the function correctly handles a full input image
    """
    full_image = np.ones((5, 5, 5))

    output_image = get_largest_CC(full_image)

    assert np.array_equal(output_image, full_image), "Expected output to be the ssame as input for full input image"


def test_get_largest_CC_no_segmentation():
        """
        Aim: check if it raises a ValueError when input image is not segmented
        """
        
        not_segmented_image = np.array([[[1, 3], [2, 0], [3, 1]], [[2, 2], [1, 1], [2, 3]], [[0, 1], [2, 2], [1, 1]]])

        try:
            get_largest_CC(not_segmented_image)
        except ValueError as e:
            assert str(e) == "Input image is not segmented", "Expected ValueError when input image is not segmented"


def test_get_largest_CC_no_connected_components():
    """
    Aim: Check if the function correctly handles an input image with no connected components
    """
    input_image = np.zeros((8, 8, 8))
    input_image[1, 1, 1] = 1
    input_image[5, 5, 5] = 1
    input_image[7, 7, 7] = 1

    try:
        get_largest_CC(input_image)
    except ValueError as e:
        assert str(e) == "No connected components found in the input image", "Expected ValueError when the image has no connected components"










#JACCARD_INDEX

def test_jaccard_output():
    """
    Aim: check that the function works properly in three easy cases
    """

    # Completely different
    seg1 = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])

    assert jaccard_index(seg1, seg2) == 0.0, "Expected Jaccard Index to be 0 when segmentations don't intersect"

    #Indentical
    seg3 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])
    seg4 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])

    assert jaccard_index(seg3, seg4) == 1.0, "Expected Jaccard Index to be 1 when segmentations are identical"

    # Partial superposition
    seg5 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])
    seg6 = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 0]]])

    assert jaccard_index(seg5, seg6) == (2/3), "Expected Jaccard Index to be 2/3"



def test_jaccard_large_volumes():
    """
    Aim: check if the function is able to deal with large volumes
    """
    #large volumes
    seg1 = np.ones((300, 300, 300))
    seg2 = np.ones((300, 300, 300))

    assert jaccard_index(seg1, seg2) == 1.0, "Expected Jaccard Index to be 1 for identical segmentations"

    #large volumes with one pixel of difference
    seg3 = np.ones((300, 300, 300))
    seg4 = np.copy(seg3)
    
    seg4[50, 50, 50] = 0

    assert isclose(jaccard_index(seg3, seg4), (seg3.size - 1) / seg3.size, rel_tol=1e-6)



def test_jaccard_complex_shapes():
    """
    Aim: test if the function is able to deal with complex volumes (spheres as approximations of brains)
    """
    seg1 = np.zeros((100, 100, 100))
    seg2 = np.zeros((100, 100, 100))

    # Create a sphere in the first segmentation
    y, x, z = np.ogrid[-50:50, -50:50, -50:50]
    mask = x**2 + y**2 + z**2 <= 30**2
    
    seg1[mask] = 1

    # Create a sphere in the second segmentation
    y, x, z = np.ogrid[-50:50, -50:50, -50:50]
    mask = (x-5)**2 + (y-5)**2 + (z-5)**2 <= 20**2

    seg2[mask] = 1


    intersection_volume = np.sum(np.minimum(seg1, seg2))
    union_volume = np.sum(np.maximum(seg1, seg2))
    
    expected_jaccard = intersection_volume / union_volume

    assert jaccard_index(seg1, seg2) == expected_jaccard





def test_jaccard_different_shapes():
    """
    Aim: test if the function raises a ValueError when input are of different shapes
    """
    seg_1 = np.array([1, 0, 1, 0, 1])
    seg_2 = np.array([1, 1, 0, 0, 1, 1])

    try:
        jaccard_index(seg_1, seg_2)
    except ValueError as e:
        assert str(e) == "Segmentations are of different shape", "Expected a ValueError when the segmentations are of different shape"





def test_jaccard_empty_segmentation():
    """
    Aim: check that the function raises an error when passing one or two void segmentations
    """
   
    seg1 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[1, 1], [1, 0]], [[1, 0], [1, 1]]])

    #one void segmentation
    try:
        jaccard_index(seg1, seg2)
    except ValueError as e:
        assert str(e) == "One of the segmentations is empty. Jaccard Index cannot be computed", "Expected a ValueError when one of the segmentations is empty"

    try:
        jaccard_index(seg2, seg1)
    except ValueError as e:
        assert str(e) == "One of the segmentations is empty. Jaccard Index cannot be computed", "Expected a ValueError when one of the segmentations is empty"

    #both void segmentations
    try:
        jaccard_index(seg1, seg1)
    except ValueError as e:
        assert str(e) == "Both of the segmentations are empty. Jaccard Index cannot be computed", "Expected a ValueError when both of the segmentations are empty"







def test_jaccard_no_segmentations():
    """
    Aim: test that the function raises a ValueError if input volumes are not segmented
    """

    seg1 = np.array([1, 0, 2, 0, 1])
    seg2 = np.array([1, 3, 0, 2, 1])

    try:
        jaccard_index(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1", "Expected a ValueError when values in input volumes are different that 0 or 1"


    # test with negative values
    seg1 = np.array([1, 0, -1, 0, 1])
    seg2 = np.array([1, 0, 0, 0, 1])

    try:
        jaccard_index(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1", "Expected a ValueError when values in input volumes are negative"








#VOLUMETRIC DIFFERENCE
def test_volumetric_difference_output():
    """
    Aim: check that the function works properly in three easy cases
    """

    seg1 = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]]])

    assert volumetric_difference(seg1, seg2) == 0, "Expected volumetric difference to be 0 if the volumes are equal"

    #identical
    assert volumetric_difference(seg2, seg2) == 0, "Expected Volumetric Difference to be 0 for identical segmentations"

    # Partial superposition
    seg3 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])
    seg4 = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 0]]])

    assert volumetric_difference(seg3, seg4) == -1




def test_volumetric_difference_large_volumes():
    """
    Aim: check if the function is able to deal with large volumes
    """
    #large volumes
    seg_1 = np.ones((300, 300, 300))
    seg_2 = np.ones((300, 300, 300))

    assert volumetric_difference(seg_1, seg_2) == 0, "Expected Volumetric Difference to be 0 for identical volumes"

    #large volumes with one pixel of difference
    seg_2[50, 50, 50] = 0

    assert volumetric_difference(seg_1, seg_2) == 1, "Expected volumetric difference to be 1 if volumes differ of one pixel only"




def test_volumetric_difference_complex_shapes():
    """
    Aim: test if the function is able to deal with complex volumes (spheres as approximations of brains)
    """
    seg1 = np.zeros((100, 100, 100))
    seg2 = np.zeros((100, 100, 100))

    y, x, z = np.ogrid[-50:50, -50:50, -50:50]
   
    mask_1 = x**2 + y**2 + z**2 <= 30**2
    seg1[mask_1] = 1

    mask_2 = (x-5)**2 + (y-5)**2 + (z-5)**2 <= 20**2
    seg2[mask_2] = 1

    #Volumetric difference should be the difference of the volumes of the two spheres
    assert isclose(volumetric_difference(seg1, seg2), (4/3)*pi*(30**3 - 20**3), rel_tol=1e-2), "Failed to compute the volumetric difference between the spheres correctly"



