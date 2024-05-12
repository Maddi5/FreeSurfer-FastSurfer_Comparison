#!/usr/bin/env python3

import numpy as np
import nibabel as nib
import pytest

from segmentations_comparison import load_volume, get_largest_CC




#LOAD_VOLUME

def test_load_volume_type_shape():
    """
    Aim: check type and shape of the output
    """
 
    assert isinstance(load_volume('Simulated_Data/Simulated_FreeSurfer.mgz'), np.ndarray)

    assert load_volume('Simulated_Data/Simulated_FreeSurfer.mgz').shape == (256, 256, 256)



def test_load_volume_segmented():
    """
    Aim: check that the output image is segmented
    """

    assert np.all((load_volume('Simulated_Data/Simulated_FreeSurfer.mgz') == 0) | (load_volume('Simulated_Data/Simulated_FreeSurfer.mgz') == 1))



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

        assert isinstance(output_image, np.ndarray)

        assert output_image.shape == input_image.shape



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

    assert np.array_equal(output_image, expected_output)


def test_get_largest_CC_single_connected_component():
    """
    Aim: Check if the function correctly handles an input image with a single connected component
    """

    #Create image with a single connected component
    input_image = np.zeros((10, 10, 10))
    input_image[3:7, 3:7, 3:7] = 1

    output_image = get_largest_CC(input_image)

    assert np.array_equal(output_image, input_image)



def test_get_largest_CC_full_input():
    """
    Aim: Check if the function correctly handles a full input image
    """
    full_image = np.ones((5, 5, 5))

    output_image = get_largest_CC(full_image)

    assert np.array_equal(output_image, full_image)


def test_get_largest_CC_no_segmentation():
        """
        Aim: check if it raises a ValueError when input image is not segmented
        """
        
        not_segmented_image = np.array([[[1, 3], [2, 0], [3, 1]], [[2, 2], [1, 1], [2, 3]], [[0, 1], [2, 2], [1, 1]]])

        try:
            get_largest_CC(not_segmented_image)
        except ValueError as e:
            assert str(e) == "Input image is not segmented"


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
        assert str(e) == "No connected components found in the input image"
