#!/usr/bin/env python3

"""

Script containing the tests for all the functions

Author: Maddalena Cavallo
Date: May 2024

"""

import pytest
import numpy as np
from math import isclose, pi

from load_functions import load_volume, get_largest_CC
from metrics import jaccard_index, volumetric_difference, Hausdorff_distance




#LOAD_VOLUME

def test_load_volume_type():

    """
    Test that load_volume() returns the correct type

    GIVEN: A valid .mgz file path.
    WHEN: The load_volume function is called on the file in this path.
    THEN: The function returns a numpy.ndarray

    """
    volume = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')

    assert isinstance(volume, np.ndarray), "Expected output type to be numpy.ndarray"


def test_load_volume_shape():

    """
    Test that load_volume() returns the correct shape.

    GIVEN: A valid .mgz file path.
    WHEN: The load_volume function is called on the file in this path.
    THEN: The function returns a numpy.ndarray with the shape (256, 256, 256).

    """
    volume = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')
    expected_shape = (256, 256, 256)
    
    assert volume.shape == expected_shape, "Expected loaded volume to have shape [256, 256, 256]"


def test_load_volume_binary():

    """
    Test that load_volume() returns a binary volume.

    GIVEN: A valid .mgz file path.
    WHEN: The load_volume function is called on the file in this path.
    THEN: The resulting volume is binary (containing only 0s and 1s).

    """
    volume = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')
    is_zero =  (volume == 0)
    is_one = (volume == 1)
    binary = np.all( is_zero | is_one)
    
    assert binary, "Expected output volume to be binary"



def test_load_volume_transpose():

    """
    Test that load_volume() correctly transposes the volume.

    GIVEN: The .mgz file path of a test volume.
    WHEN: The load_volume function is called on this file.
    THEN: The resulting volume matches the expected transposed output.

    """

    loaded_volume = load_volume('Test_volumes/test_input_volume.mgz')

    #transposed and binarized input image
    expected_output = np.array([[[1, 0], [0, 1]],
                            [[1, 1], [1, 1]]])

    assert np.array_equal(loaded_volume, expected_output), "Failed to correctly transpose the volume"



def test_load_volume_not_existent():

    """
    Test that load_volume() raises a FileNotFoundError when the filename doesn't exist

    GIVEN: An invalid .mgz file path.
    WHEN: The load_volume function is called with this path.
    THEN: A FileNotFoundError is raised with the appropriate message.

    """

    with pytest.raises(FileNotFoundError, match="File non_existent_file.mgz not found"):
        load_volume('non_existent_file.mgz')


def test_load_volume_empty_image():

    """
    Test that load_volume() raises a ValueError for an empty input image.

    GIVEN: A .mgz file path of an empty volume.
    WHEN: The load_volume function is called with this path.
    THEN: A ValueError is raised with the appropriate message.

    """

    with pytest.raises(ValueError, match = "The input image contains only null values"):
        load_volume('Test_volumes/empty_volume.mgz')



#GET_LARGEST_CC

def test_get_largest_CC_type():

    """
    Test that get_largest_CC() returns the correct type

    GIVEN: A valid output volume from load_volume function.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function returns a numpy.ndarray

    """
    input_image = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')
    
    output_image = get_largest_CC(input_image)

    assert isinstance(output_image, np.ndarray), "Expected output type to be numpy.ndarray"



def test_get_largest_CC_shape():

    """
    Test that get_largest_CC() returns the correct shape.

    GIVEN: A valid output volume from load_volume function.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function returns a numpy.ndarray with the same shape as the input

    """
    input_image = load_volume('Simulated_Data/Simulated_FreeSurfer.mgz')
    
    output_image = get_largest_CC(input_image)

    assert output_image.shape == input_image.shape, "Expected output shape to be the same as input"



def test_get_largest_CC_check_largest():

    """
    Test that get_largest_CC() correctly identifies the largest connected component.

    GIVEN: A binary volume with two connected components.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function returns a volume with only the largest connected component among the two.

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
    Test that get_largest_CC() correctly handles an input image with a single connected component.

    GIVEN: A binary volume with a single connected component.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function returns a volume identical to the input.

    """

    #Create image with a single connected component
    input_image = np.zeros((10, 10, 10))
    input_image[3:7, 3:7, 3:7] = 1

    output_image = get_largest_CC(input_image)

    assert np.array_equal(output_image, input_image), "Expected output to be the same as input for single connected component"



def test_get_largest_CC_full_input():

    """
    Test that get_largest_CC() correctly handles a full input image.

    GIVEN: An input volume where all elements are 1.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function returns a volume identical to the input.

    """
    full_image = np.ones((5, 5, 5))

    output_image = get_largest_CC(full_image)

    assert np.array_equal(output_image, full_image), "Expected output to be the same as input for full input image"


def test_get_largest_CC_not_binary():

    """
    Test that get_largest_CC raises a ValueError for a non-binary input image.

    GIVEN: A non-binary input volume.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function raises a ValueError with the correct error message.
    """

    not_binary_image = np.array([[[1, 3], [2, 0], [3, 1]], [[2, 2], [1, 1], [2, 3]], [[0, 1], [2, 2], [1, 1]]])

    with pytest.raises(ValueError, match = "Input image is not binary"):
        get_largest_CC(not_binary_image)


def test_get_largest_CC_no_connected_components():

    """  
    Test that get_largest_CC() raises a ValueError with an input image with no connected components.

    GIVEN: A binary volume with no connected components.
    WHEN: The get_largest_CC function is called with this volume.
    THEN: The function raises a ValueError with the correct error message.
    """

    input_image = np.zeros((8, 8, 8))

    with pytest.raises(ValueError, match = "No connected components found in the input image"):
        get_largest_CC(input_image)








#JACCARD_INDEX

def test_jaccard_output_no_intersection():

    """
    Test the jaccard_index function when volumes don't intersect

    GIVEN: Two binary volumes without intersection
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index

    """

    seg1 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])
    seg2 = np.array([[[0, 0], [0, 1]], [[0, 0], [0, 1]]])

    observed = jaccard_index(seg1, seg2)
    expected = 0.0

    assert observed == expected, "Expected Jaccard Index to be 0 when segmentations don't intersect"



def test_jaccard_output_identical():

    """
    Test the jaccard_index function when volumes are identical

    GIVEN: Two identical binary volumes
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index

    """

    seg1 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])

    observed = jaccard_index(seg1, seg1)
    expected = 1.0

    assert observed == expected, "Expected Jaccard Index to be 1 when segmentations are identical"



def test_jaccard_output_partial_intersection():

    """
    Test the jaccard_index function when the volumes partially intersect

    GIVEN: Two binary volumes
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index
    """

    seg1 = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 0]]])
    seg2 = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 0]]])

    observed = jaccard_index(seg1, seg2)
    expected = 2/3

    assert  observed == expected, "Expected Jaccard Index to be 2/3"




def test_jaccard_large_volumes_identical():

    """
    Test the jaccard_index function with large volumes.

    GIVEN: Two large binary volumes for comparison.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index, even for large volumes.

    """
    #large volumes
    seg1 = np.ones((300, 300, 300))

    expected = 1.0

    assert jaccard_index(seg1, seg1) == expected, "Expected Jaccard Index to be 1 for identical segmentations"

 


def test_jaccard_large_volumes_one_pixel():

    """
    Test the jaccard_index function with large volumes that differ for one pixel

    GIVEN: Two large binary volumes with one different pixel for comparison.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index, even for large volumes.

    """
    #large volumes
    seg1 = np.ones((300, 300, 300))
    seg2 = np.ones((300, 300, 300))

    #one pixel of difference
    seg2[50, 50, 50] = 0

    observed = jaccard_index(seg1, seg2)
    expected = (seg1.size - 1) / seg1.size
    
    assert isclose(observed, expected, rel_tol=1e-6)





def test_jaccard_complex_shapes():

    """
    Test the jaccard_index function with complex shapes (cubic shapes)

    GIVEN: Two binary volumes with complex shapes (cubic).
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function returns the correct Jaccard Index for these complex shapes.

    """

    seg1 = np.zeros((100, 100, 100))
    seg2 = np.zeros((100, 100, 100))

    # Create a cube in the first volume
    seg1[20:50, 20:50, 20:50] = 1

    # Create a cube in the second volume
    seg2[30:60, 30:60, 30:60] = 1
    
    observed_jaccard = jaccard_index(seg1, seg2)

    intersection_expected = 20*20*20
    union_expected = (30*30*30*2) - intersection_expected
    expected_jaccard = intersection_expected / union_expected


    assert observed_jaccard == expected_jaccard






def test_jaccard_different_shapes():

    """
    Test that the jaccard_index function raises a ValueError when inputs are of different shapes.

    GIVEN: Two volumes of different shapes.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function raises a ValueError indicating that the shapes are different.

    """

    seg1 = np.zeros((5, 5, 5))
    seg2 = np.zeros((6, 6, 6))

    with pytest.raises(ValueError, match = "Input volumes are of different shape"):
        jaccard_index(seg1, seg2)


def test_jaccard_one_empty_volume():

    """
    Test that the Jaccard Index function raises a ValueError for one empty volume.

    GIVEN: Binary volumes with one being empty.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function raises a ValueError indicating that one segmentation is empty.

    """
   
    seg1 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[1, 1], [1, 0]], [[1, 0], [1, 1]]])

    with pytest.raises(ValueError, match = "One of the segmentations is empty. Jaccard Index cannot be computed"):
        jaccard_index(seg1, seg2)


  

def test_jaccard_empty_volumes():

    """
    Test that the Jaccard Index function raises a ValueError for two empty volumes.

    GIVEN: Binary volumes with both being empty.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function raises a ValueError indicating that the segmentations are empty.

    """
   
    seg1 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])

    with pytest.raises(ValueError, match = "Both of the segmentations are empty. Jaccard Index cannot be computed"):
        jaccard_index(seg1, seg1)



def test_jaccard_not_binary():

    """
    Test that the jaccard_index function raises a ValueError if input volumes are non-binary.

    GIVEN: Volumes that contain values other than 0 and 1.
    WHEN: The jaccard_index function is called with these volumes.
    THEN: The function raises a ValueError indicating that the volumes are not binary.

    """

    seg1 = np.array([[[1, 0], [1, 1]], [[1, 0], [1, 0]]])
    seg2 = np.array([[[1, 3], [0, 2]], [[1, 0], [0, 0]]])
    seg3 = np.array([[[1, 0], [-1, 0]], [[1, 0], [1, 0]]])

    with pytest.raises(ValueError, match = "Input volumes contain values other than 0 and 1"):
        jaccard_index(seg1, seg2)

    with pytest.raises(ValueError, match = "Input volumes contain values other than 0 and 1"):
        jaccard_index(seg1, seg3)






#VOLUMETRIC DIFFERENCE

def test_volumetric_difference_output_equal_volume():

    """
    Test the volumetric_difference function with segmentations with same volumes

    GIVEN: The binary segmentations with identical volumes for comparison.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference

    """

    seg1 = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]]])

    observed = volumetric_difference(seg1, seg2)
    expected = 0

    assert observed == expected, "Expected volumetric difference to be 0 if the volumes are equal"

def test_volumetric_difference_output_identical():

    """
    Test the volumetric_difference function with identical input

    GIVEN: Two identical binary volumes for comparison.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference

    """

    seg1 = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]]])

    observed = volumetric_difference(seg1, seg1)
    expected = 0

    assert observed == expected, "Expected Volumetric Difference to be 0 for identical segmentations"


def test_volumetric_difference_output_one_pixel():

    """
    Test the volumetric_difference function with volumes with one pixel difference

    GIVEN: The binary volumes for comparison with one different pixel
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference 

    """

    seg1 = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[1, 0], [0, 0]], [[1, 0], [0, 1]]])


    observed = volumetric_difference(seg1, seg2)
    expected = -1

    assert observed == expected, "Expected Volumetric Difference to be -1"




def test_volumetric_difference_large_volumes_identical():

    """
    Test the volumetric_difference function with identical large volumes.

    GIVEN: Two large identical binary volumes for comparison.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference, even for large volumes
    """

    #large volumes
    seg1 = np.ones((300, 300, 300))

    observed = volumetric_difference(seg1, seg1)
    expected = 0

    assert observed == expected, "Expected Volumetric Difference to be 0 for identical volumes"


def test_volumetric_difference_large_volumes_one_pixel():

    """
    Test the volumetric_difference function with large volumes that differ of one pixel only

    GIVEN: Two large binary volumes with only one different pixel for comparison.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference, even for large volumes.
    """

    #large volumes
    seg1 = np.ones((300, 300, 300))
    seg2 = np.ones((300, 300, 300))

    #one pixel of difference
    seg2[50, 50, 50] = 0

    observed = volumetric_difference(seg1, seg2)
    expected = 1

    assert observed == expected, "Expected volumetric difference to be 1 if volumes differ of one pixel only"



def test_volumetric_difference_complex_shapes():

    """
    Test the volumetric_difference function with complex shapes (spheres as approximations of brains).

    GIVEN: Two binary volumes with complex shapes (spheres).
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function returns the correct volumetric difference for these complex shapes.
    """

    seg1 = np.zeros((100, 100, 100))
    seg2 = np.zeros((100, 100, 100))

    y, x, z = np.ogrid[-50:50, -50:50, -50:50]
   
    mask_1 = x**2 + y**2 + z**2 <= 30**2
    seg1[mask_1] = 1

    mask_2 = (x-5)**2 + (y-5)**2 + (z-5)**2 <= 20**2
    seg2[mask_2] = 1

    observed = volumetric_difference(seg1, seg2)
    #Volumetric difference should be the difference of the volumes of the two spheres
    expected = (4/3)*pi*(30**3 - 20**3)

    assert isclose(observed, expected, rel_tol=1e-2), "Failed to compute the volumetric difference between the spheres correctly"



def test_volumetric_difference_empty_volumes():

    """
    Test that the volumetric_difference function raises a ValueError for empty input volumes.

    GIVEN: Binary volumes with one or both being empty.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function raises a ValueError indicating the volume(s) is/are empty.
    """

    seg1 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[1, 0], [0, 1]], [[1, 0], [1, 1]]])

    #one empty segmentation
    try:
        volumetric_difference(seg1, seg2)
    except ValueError as e:
        assert str(e) == "One of the segmented volumes is empty. Check input data", "Expected a ValueError when one of the segmentations is empty"

    try:
        volumetric_difference(seg2, seg1)
    except ValueError as e:
        assert str(e) == "One of the segmented volumes is empty. Check input data", "Expected a ValueError when one of the segmentations is empty"


    #both empty segmentations
    try:
        volumetric_difference(seg1, seg1)
    except ValueError as e:
        assert str(e) == "Both of the segmented volumes are empty. Check input data", "Expected a ValueError when both of the segmentations are empty"




def test_volumetric_difference_different_shapes():

    """
    Test that the volumetric_difference function raises a ValueError for inputs with different shapes.

    GIVEN: Two volumes of different shapes.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function raises a ValueError indicating the shapes are different.

    """

    seg1 = np.zeros((5, 5, 5))
    seg2 = np.zeros((6, 6, 6))

    try:
        volumetric_difference(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input volumes are of different shape", "Expected a ValueError when the input volumes are of different shape"



def test_volumetric_difference_not_binary():

    """
    Test that the volumetric_difference function raises a ValueError when input data are non-binary.

    GIVEN: Volumes that contain values other than 0 and 1.
    WHEN: The volumetric_difference function is called with these volumes.
    THEN: The function raises a ValueError indicating the volumes are not binary.

    """

    seg1 = np.array([[[1, 0], [1, 1]], [[1, 0], [1, 0]]])
    seg2 = np.array([[[1, 3], [0, 2]], [[1, 0], [0, 0]]])
    seg3 = np.array([[[1, 0], [-1, 0]], [[1, 0], [1, 0]]])

    try:
        volumetric_difference(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1. Check input data", "Expected a ValueError when values in the volumes are not binary"

    try:
        volumetric_difference(seg1, seg3)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1. Check input data", "Expected a ValueError when values in the volumes are negative"





#HAUSDORFF DISTANCE

def test_Hausdorff_2D():

    """
    Test the Hausdorff_distance function with 2D data.

    GIVEN: Two binary 2D volumes for comparison.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function returns the correct Hausdorff distance.

    """

    seg1 = np.array([[1, 1], [0, 0]])
    seg2 = np.array([[1, 0], [0, 1]])

    Hausdorff_dist_obs = Hausdorff_distance(seg1, seg2)

    expected = 0.5

    assert Hausdorff_dist_obs == expected, "Expected Hausdorff distance to be 0.5"


def test_Hausdorff_3D_identical():

    """
    Test the Hausdorff_distance function with identical 3D volumes.

    GIVEN: Two identical binary 3D volumes for comparison.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function returns a Hausdorff distance of 0.

    """
    
    seg1 = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 0]]])

    observed = Hausdorff_distance(seg1, seg1)
    expected = 0

    assert observed == expected, "Expected Hausdorff distance to be 0 for identical segmentations"


def test_Hausdorff_3D_single_point_segmentations():

    """
    Test the Hausdorff_distance function with simple 3D data.

    GIVEN: Two binary 3D volumes each containing a single non-zero point.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function returns the correct Hausdorff distance between the points.
    """

    seg1 = np.zeros((3, 3, 3))
    seg2 = np.zeros((3, 3, 3))

    seg1[0, 0, 0] = 1
    seg2[1, 1, 1] = 1

    observed = Hausdorff_distance(seg1, seg2)
    expected =  np.sqrt(3)

    assert observed == expected, "Expected Hausdorff distance to be sqrt(3)"




def test_Hausdorff_3D_large_volumes():

    """    
    Test the Hausdorff_distance function with large volumes.

    GIVEN: Two large 3D volumes for comparison.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function returns the correct Hausdorff distance, even for large volumes.
    """

    seg1 = np.ones((300, 300, 300))
    seg2 = np.ones((300, 300, 300))

    expected = 0.0
    
    assert Hausdorff_distance(seg1, seg2) == expected, "Expected Hausdorff distance to be 0 for identical segmentations"



def test_Hausdorff_empty_volume():

    """
    Test that the Hausdorff_distance function raises a ValueError for empty input volumes.

    GIVEN: Binary volumes with one or both being empty.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function raises a ValueError indicating the volume(s) is/are empty.
    """

    seg1 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    seg2 = np.array([[[1, 1], [0, 1]], [[1, 0], [1, 1]]])

    #first segmentation is empty
    try:
        Hausdorff_distance(seg1, seg2)
    except ValueError as e:
        assert str(e) == "One of the segmentations is empty. Hausdorff distance cannot be computed", "Expected a ValueError when one of the segmentations is empty"

    #second segmentation is empty
    try:
        Hausdorff_distance(seg2, seg1)
    except ValueError as e:
        assert str(e) == "One of the segmentations is empty. Hausdorff distance cannot be computed", "Expected a ValueError when one of the segmentations is empty"

    #both segmentations are empty
    try:
        Hausdorff_distance(seg1, seg1)
    except ValueError as e:
        assert str(e) == "Both of the segmentations are empty. Hausdorff distance cannot be computed", "Expected a ValueError when both the segmentations are empty"






def test_Hausdorff_different_shapes():

    """
    Test that the Hausdorff_distance function raises a ValueError for inputs with different shapes.

    GIVEN: Two volumes of different shapes.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function raises a ValueError indicating that the shapes are different.
    """

    seg1 = np.zeros((300, 300, 300))
    seg2 = np.zeros((200, 200, 200))

    try:
        Hausdorff_distance(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input data are of different shape. Hausdorff distance cannot be computed", "Expected ValueError when input data are of different shape"



def test_Hausdorff_not_binary():
    
    """
    Test that the Hausdorff_distance function raises a ValueError when input data are non-binary.

    GIVEN: Volumes that contain values other than 0 and 1.
    WHEN: The Hausdorff_distance function is called with these volumes.
    THEN: The function raises a ValueError indicating the volumes are not binary.
    """

    seg1 = np.array([1, 0, 0, 0, 1])
    seg2 = np.array([1, 0, 2, 0, 1])
    seg3 = np.array([1, 0, -1, 0, 1])

    try:
        Hausdorff_distance(seg1, seg2)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1. Check input data", "Expected a ValueError when values in input volume are different from 0 or 1"


    try:
        Hausdorff_distance(seg1, seg3)
    except ValueError as e:
        assert str(e) == "Input volumes contain values other than 0 and 1. Check input data", "Expected a ValueError when values in input volume are negative"

