#!/usr/bin/env python3

"""

Script to load MRI images from .mgz files, compute the metrics for the comparison and analyze the difference matrix.

Author: Maddalena Cavallo
Date: May 2024

"""


import numpy as np
import pandas as pd
from medpy.metric.binary import dc

from load_functions import load_volume, get_largest_CC
from metrics import jaccard_index, volumetric_difference, Hausdorff_distance



# Load volumes
print("\nLoading volumes...\n")

input_volumes= {
    'FreeSurfer': load_volume('Simulated_Data/Simulated_FreeSurfer.mgz'),
    'FreeSurfer_auto': load_volume('Simulated_Data/Simulated_FreeSurfer_auto.mgz'),
    'FastSurfer': load_volume('Simulated_Data/Simulated_FastSurfer.mgz')
}

# Convert to int type to be able perform the subtraction for the difference matrix
data = {
    'FreeSurfer': get_largest_CC(input_volumes['FreeSurfer']).astype(int),
    'FreeSurfer_auto': get_largest_CC(input_volumes['FreeSurfer_auto']).astype(int),
    'FastSurfer': get_largest_CC(input_volumes['FastSurfer']).astype(int)
}


#Define cases of comparison
cases = [('FreeSurfer', 'FreeSurfer_auto'), ('FreeSurfer', 'FastSurfer'), ('FreeSurfer_auto', 'FastSurfer')]


print("\nComputing the metrics for each case of comparison...")

results_list = []

#Compute the metrics in each case of comparison
for case in cases:

    dice_coeff = dc(data[case[0]], data[case[1]])
    jaccard_ind = jaccard_index(data[case[0]], data[case[1]])
    vol_difference = volumetric_difference(data[case[0]], data[case[1]])
    hausdorff_dist = Hausdorff_distance(data[case[0]], data[case[1]])

    print(f"\nCase {case[0]} vs {case[1]}:")
    print("Dice Similarity Coefficient:", dice_coeff)
    print("Jaccard index:", jaccard_ind)
    print("Volumetric difference:", vol_difference)
    print("Hausdorff Distance:", hausdorff_dist)

    results_list.append([f'Case {case[0]} vs {case[1]}', dice_coeff, jaccard_ind, vol_difference, hausdorff_dist])


dataframe_metrics = pd.DataFrame(results_list, columns=['Case of comparison', 'Dice Similarity Coefficient', 'Jaccard Index', 'Volumetric difference', "Hausdorff Distance"])
dataframe_metrics.to_excel('Results/results_metrics.xlsx', index=False)



# Difference matrix
print()
print("\nComputing difference matrix...\n")

for case in cases:

    print(f"Case {case[0]} vs {case[1]}:")

    data1, data2 = data[case[0]], data[case[1]]

    difference_matrix = data1 - data2  # 3D matrix made of 0, 1 e -1
    
    np.save(f'Results/difference_matrix_{case[0]}_vs_{case[1]}.npy', difference_matrix)


    #on the entire volume

    #Compute mean, standard deviation, total sum
    mean = np.mean(difference_matrix)
    standard_deviation = np.std(difference_matrix)
    total_sum = np.sum(difference_matrix)

    #To give a general idea to the user, not to be saved
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {standard_deviation}")
    print(f"Total Sum: {total_sum}")
    print()


    #slice by slice

    #iterate over the three planes
    for plane, axis in [('Axial', 0), ('Coronal', 1), ('Sagittal', 2)]:

        print(f"{plane} plane")

        differences = []


        #Iterate over the slices
        for i in range(difference_matrix.shape[axis]):

            #select in which direction to move to "cut" the brain
            if axis == 0:
                slice = difference_matrix[i, :, :]
            elif axis == 1:
                slice = difference_matrix[:, i, :]
            elif axis == 2:
                slice = difference_matrix[:, :, i]


            # Sum of the difference matrix in that slice
            slice_sum = np.sum(slice)

            #Apply the absolute value to compute the total number of different pixels in that slice
            slice_num_diff = np.sum(np.abs(slice))

            slice_total_pixels = slice.shape[0]*slice.shape[1] # take always the same axis because it's symmetric in 3D

            # Compute the percentage of different pixels in that slice
            perc_diff = (slice_num_diff / slice_total_pixels) * 100


            #Add all the results to the list for that slice
            differences.append([i, slice_num_diff, perc_diff, slice_sum])


        # Save the results for all the slices in a dataframe
        print(f"Saving dataframe for {plane} view\n")

        dataframe_slices = pd.DataFrame(differences, columns = ["Slice", "# different pixels", "Percentage difference", "Total sum"])
        dataframe_slices.to_excel(f"Results/Differences_{case[0]}_vs_{case[1]}_{plane}.xlsx", index=False)


