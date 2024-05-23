import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def create_bar_plot(data, view):

    fig = plt.figure(figsize=(10,7))

    for i, dataframe in enumerate(data):

        plt.subplot(2, 2, i+1)

        sns.barplot(x = "Slice", y = "Total sum", data = dataframe)

        plt.xlabel('Slice')
        plt.ylabel('Total sum of differences')
        plt.title(f'{cases[i][0]} vs {cases[i][1]}')

        non_zero_indices = dataframe[dataframe['Total sum'] != 0].index
        x_min = round(dataframe.loc[non_zero_indices[0], 'Slice'], -1)
        x_max = round(dataframe.loc[non_zero_indices[-1], 'Slice'], -1)
        plt.xlim(x_min-10, x_max+10)
        plt.xticks(range(x_min-10, x_max+10, 10))

        y_min = round(int(dataframe['Total sum'].min()), -1)
        y_max = round(int(dataframe['Total sum'].max()), -1)
        plt.yticks(range(y_min, y_max, 100))

    # Captions dimensions
    plt.rc('axes', titlesize=9)    # title
    plt.rc('axes', labelsize=7)    # axis labels
    plt.rc('xtick', labelsize=6)    # Labels on x
    plt.rc('ytick', labelsize=6)    # Labels on y

    fig.suptitle(f'{view} view', fontsize=12)

    print("Saving the plot...")
    plt.savefig(f'Results/Total_sum_differences_{view}_view.png')

    plt.show()



def create_line_plot(data, view):

    plt.figure(figsize=(10, 7))

    for i, dataframe in enumerate(data):

        sns.lineplot(x = "Slice", y = "Percentage difference", data=dataframe, label = f'Case {cases[i][0]} vs {cases[i][1]}')

    plt.xlabel('Slice')
    plt.ylabel('% of different pixels')

    non_zero_indices = dataframe[dataframe['Percentage difference'] != 0].index
    x_min = round(dataframe.loc[non_zero_indices[0], 'Slice'], -1)
    x_max = round(dataframe.loc[non_zero_indices[-1], 'Slice'], -1)
    plt.xlim(x_min-20, x_max+20)
    plt.xticks(range(x_min-20, x_max+20, 10))

    plt.title(f'{view} view')

    #Captions dimensions
    plt.rc('axes', labelsize=15)    # title
    plt.rc('xtick', labelsize=8)    # Labels on x
    plt.rc('ytick', labelsize=8)    # Labels on y
    plt.rc('legend', fontsize=8)    # Legend dimension



    print("Saving the plot...")
    plt.savefig(f'Results/% Different_pixel_{view}.png')

    plt.show()






def create_3d_plot(difference_matrix, view_angle, output_filename, dpi):

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(projection='3d')

    # Find coordinates of negative points
    x1, y1, z1 = np.where(difference_matrix == -1)
    # Find coordinates of positive points
    x2, y2, z2 = np.where(difference_matrix == 1)

    #plot negative points in blue
    ax.plot(x1, y1, z1, 'b.', markersize=0.1, label = "FastSurfer")
    #plot positive points in red
    ax.plot(x2, y2, z2, 'r.', markersize=0.1, label = "FreeSurfer")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0, 256)

    ax.set_xticks(np.arange(0, 255, 50))
    ax.set_yticks(np.arange(0, 255, 50))
    ax.set_zticks(np.arange(0, 255, 250))

    #Set elevation and azimuth angles
    ax.view_init(*view_angle)

    plt.savefig(output_filename, dpi = dpi)

    plt.tight_layout()
    plt.show()






#Define cases of comparison
cases = [('FreeSurfer', 'FreeSurfer_auto'), ('FreeSurfer', 'FastSurfer'), ('FreeSurfer_auto', 'FastSurfer')]




print("Loading the data from the files...")

#Create lists of dataframes
data_axial = [pd.read_excel(f"Results/Differences_{case[0]}_vs_{case[1]}_Axial.xlsx") for case in cases]
data_coronal = [pd.read_excel(f"Results/Differences_{case[0]}_vs_{case[1]}_Coronal.xlsx") for case in cases]
data_sagittal = [pd.read_excel(f"Results/Differences_{case[0]}_vs_{case[1]}_Sagittal.xlsx") for case in cases]





# Bar plots
print("\nComputing bar plots...")

# Axial
print("Axial view")
create_bar_plot(data_axial, 'Axial')

# Coronal
print("Coronal view")
create_bar_plot(data_coronal, 'Coronal')

# Sagittal
print("Sagittal view")
create_bar_plot(data_sagittal, 'Sagittal')





#Line plots
print("\nComputing line plots...")

# Axial
print("Axial view")
create_line_plot(data_axial, 'Axial')

# Coronal
print("Coronal view")
create_line_plot(data_coronal, 'Coronal')

# Sagittal
print("Sagittal view")
create_line_plot(data_sagittal, 'Sagittal')








#Difference matrix 2D
print("\nVisualising difference matrices...")

difference_matrices = []

for case in cases:
    difference_matrix = np.load(f'Results/difference_matrix_{case[0]}_vs_{case[1]}.npy')
    difference_matrices.append(difference_matrix)


#Ask the user which slices to visualise
while True:
    try:
        print("\nSelect which slices you want to visualise")

        first = int(input("Enter the start of the slice range (>=0): "))
        last = int(input("Enter the end of the slice range (<256): "))
        step = int(input("Enter the step for the slice range (>0): "))

        break

    except ValueError:
        print("Invalid input. Please enter an integer.")


#set the slices to see
slice_index = list(range(first, last, step))

case_names = ['A', 'B', 'C']
projection_names = ['Axial', 'Coronal', 'Sagittal']


# For each case
for i, diff_matrix in enumerate(difference_matrices):

    print(f"\nCase {case_names[i]}")

    # For each projection
    for k, projection in enumerate(projection_names):
        print(f"{projection} view")
        
        # For each slice
        for j in slice_index:
            print(f"Slice {j}")
            #Select the correct projection
            if projection == 'Axial':
                slice_data = diff_matrix[j, :, :]
            elif projection == 'Coronal':
                slice_data = diff_matrix[:, j, :]
            elif projection == 'Sagittal':
                slice_data = diff_matrix[:, :, j]


            if np.any(slice_data != 0):

                plt.figure(figsize=(8, 6))
                plt.imshow(slice_data, cmap='coolwarm', aspect='auto')
                plt.colorbar(label='Difference')
                plt.title(f'Slice {j} {projection} plane ({case_names[i]})')
                plt.xlabel('Voxel')
                plt.ylabel('Voxel')

                save_path = f'Results/{projection}_slice_{j}_{case_names[i]}.png'
                plt.savefig(save_path)

                plt.show()







#Difference matrix 3D
print("Visualizing 3D difference matrix for FreeSurfer-FastSurfer comparison...")

# one side
create_3d_plot(difference_matrices[1], (90, 0), f'Results/Difference_matrix_3D_{cases[1][0]}_vs_{cases[1][1]}_right.png', 800)
# other side
create_3d_plot(difference_matrices[1], (-90, -180), f'Results/Difference_matrix_3D_{cases[1][0]}_vs_{cases[1][1]}_left.png', 800)
# from the bottom 
create_3d_plot(difference_matrices[1], (-125, -25), f'Results/Difference_matrix_3D_{cases[1][0]}_vs_{cases[1][1]}_bottom.png', 1500)


