import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def create_bar_plot(data, view):

    fig = plt.figure(figsize=(10,7))

    # Captions dimensions
    plt.rc('axes', titlesize=9)    # title
    plt.rc('axes', labelsize=7)    # axis labels
    plt.rc('xtick', labelsize=6)    # Labels on x
    plt.rc('ytick', labelsize=6)    # Labels on y

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


    #Captions dimensions
    plt.rc('axes', labelsize=15)    # title
    plt.rc('xtick', labelsize=8)    # Labels on x
    plt.rc('ytick', labelsize=8)    # Labels on y
    plt.rc('legend', fontsize=7)    # Legend dimension



    print("Saving the plot...")
    plt.savefig(f'Results/% Different_pixel_{view}.png')

    plt.show()
    print()



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



