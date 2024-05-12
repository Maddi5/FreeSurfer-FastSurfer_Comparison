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
 
    assert isinstance(load_volume('simulated_freesurfer.mgz'), np.ndarray)

    assert load_volume('simulated_freesurfer.mgz').shape == (256, 256, 256)



def test_load_volume_segmented():
    """
    Aim: check that the output image is segmented
    """

    assert np.all((load_volume('simulated_freesurfer.mgz') == 0) | (load_volume('simulated_freesurfer.mgz') == 1))



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
        input_image = load_volume('simulated_freesurfer.mgz')

        output_image = get_largest_CC(input_image)

        assert isinstance(output_image, np.ndarray)

        assert output_image.shape == input_image.shape


