import nibabel as nib
import numpy as np


#create a test input volume
test_input_volume = np.array([[[0, 1], [2, 4]], 
                        [[3, 0], [1, 2]]])
    
nib.save(nib.Nifti1Image(test_input_volume, np.eye(4)), 'Test_volumes/test_input_volume.mgz') 


# create a test volume with null values
empty_volume = np.zeros((256, 256, 256))
nib.save(nib.Nifti1Image(empty_volume, np.eye(4)), 'Test_volumes/empty_volume.mgz')
