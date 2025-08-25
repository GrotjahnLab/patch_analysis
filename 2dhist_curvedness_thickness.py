import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the csv of average thickness
df1 = pd.read_csv('/scratch1/users/atty/ATP_synthase_edgefiltered/average_thickness_per_patch.csv')

# Load the csv of average curvedness
df2 = pd.read_csv('/scratch1/users/atty/ATP_synthase_edgefiltered/average_curvature_per_patch.csv')

# Extract columns of tomogram, patch_number, and average_thickness_patch_per_triangle from df1
df1_patch_thickness = df1[['tomogram', 'patch_number', 'average_thickness_patch_per_triangle']]

# Extract columns of tomogram, patch_random_number, and average_thickness_random_patch_per_triangle from df1
df1_random_thickness = df1[['tomogram', 'patch_random_number', 'average_thickness_random_patch_per_triangle']]

# Merge average_curvedness_patch_per_triangle from df2 to df1_patch_thickness
df1_patch_thickness_curvedness = pd.merge(
    df1_patch_thickness, 
    df2[['tomogram', 'patch_number', 'average_curvedness_patch_per_triangle']], 
    on=['tomogram', 'patch_number']
)
print(len(df1_patch_thickness_curvedness))
# filter out nan values
df1_patch_thickness_curvedness = df1_patch_thickness_curvedness.dropna()
print(len(df1_patch_thickness_curvedness))
# only keep the rows with average_thickness_patch_per_triangle in the range of 2.5 to 4.5 
# and average_curvedness_patch_per_triangle which log value in the range of -2.4 to -1
df1_patch_thickness_curvedness = df1_patch_thickness_curvedness[
    (df1_patch_thickness_curvedness['average_thickness_patch_per_triangle'].between(2.5, 4.5)) &
    (np.log10(df1_patch_thickness_curvedness['average_curvedness_patch_per_triangle']).between(-2.4, -1))
]
print('Number of ATP synthase patches after filtering:', len(df1_patch_thickness_curvedness))


# Merge average_curvedness_random_patch_per_triangle from df2 to df1_random_thickness
df1_random_thickness_curvedness = pd.merge(
    df1_random_thickness, 
    df2[['tomogram', 'patch_random_number', 'average_curvedness_random_patch_per_triangle']], 
    on=['tomogram', 'patch_random_number']
)
print(len(df1_random_thickness_curvedness))
# filter out nan values
df1_random_thickness_curvedness = df1_random_thickness_curvedness.dropna()
print(len(df1_random_thickness_curvedness))
# only keep the rows with average_thickness_random_patch_per_triangle in the range of 2.5 to 4.5 
# and average_curvedness_random_patch_per_triangle which log10 value in the range of -2.4 to -1
df1_random_thickness_curvedness = df1_random_thickness_curvedness[
    (df1_random_thickness_curvedness['average_thickness_random_patch_per_triangle'].between(2.5, 4.5)) &
    (np.log10(df1_random_thickness_curvedness['average_curvedness_random_patch_per_triangle']).between(-2.4, -1))
]
print('Number of random patches after filtering:', len(df1_random_thickness_curvedness))

# Plot 2D histogram of average thickness and average curvedness with log scale for patches
plt.figure(figsize=(10, 6))
plt.hist2d(
    df1_patch_thickness_curvedness['average_thickness_patch_per_triangle'],
    np.log10(df1_patch_thickness_curvedness['average_curvedness_patch_per_triangle']),
    bins=20,  # Adjust the number of bins as needed
    cmap='viridis',
    vmin=0,vmax=15
)
# Add colorbar
plt.colorbar(label='Count')
# Add labels and title
#plt.xlim(2.5,4.5)
#plt.ylim(-1.8, -1)
#plt.ylim(-2.4, -1)
plt.xlabel('Thickness per patch (nm)')
plt.ylabel('Log10(Curvedness per patch)')
plt.title('Thickness v.s. Log10(Curvedness)_ATP Synthase Patches')
# Show plot
plt.tight_layout()
plt.show()

# Plot 2D histogram of average thickness and average curvedness with log scale for random patches
plt.figure(figsize=(10, 6))
plt.hist2d(
    df1_random_thickness_curvedness['average_thickness_random_patch_per_triangle'],
    np.log10(df1_random_thickness_curvedness['average_curvedness_random_patch_per_triangle']),
    bins=20,  # Adjust the number of bins as needed
    cmap='viridis',
    vmin=0,vmax=15
)
# Add colorbar
plt.colorbar(label='Count')
# Add labels and title
#plt.xlim(2.5,4.5)
#plt.ylim(-1.8, -1)
#plt.ylim(-2.4, -1)
plt.xlabel('Thickness per patch (nm)')
plt.ylabel('Log10(Curvedness per patch)')
plt.title('Thickness v.s. Log10(Curvedness)_Random Patches')
# Show plot
plt.tight_layout()
plt.show()
