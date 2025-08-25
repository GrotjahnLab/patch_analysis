import glob
import os
import click
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pycurv import TriangleGraph, io
from graph_tool import load_graph, GraphView

@click.command()
@click.option("--gt_dir", required=True, help="Directory containing the .gt files with patches")
@click.option("--output_csv", required=True, help="Path to the output CSV file")
def average_curvedness_calculation(gt_dir, output_csv):
    # Create an empty list to store the average curvedness data
    average_curvedness_data = []

    # Collect all .gt files in the specified directory
    gt_graphs = glob.glob(os.path.join(gt_dir, "*.gt"))
    

    #count the average curvature per patch for each t
    for graph in gt_graphs:
        # Load the graph
        tg = TriangleGraph()
        tg.graph = load_graph(graph)
        # Extract the tomogram name before .labels_IMM.AVV_rh8_refined_with_patches.gt
        tomogram_name = os.path.basename(graph).split('.labels_IMM.AVV_rh8_edgefiltered_refined_individual_patch.gt')[0]
        print(tomogram_name)

        # Fetch the vertex property patch_center and patch_random_center
        patch_center = tg.graph.vp.patch_center
        patch_random_center = tg.graph.vp.patch_random_center
        # find the value of patch center that is not 0
        patch_center_value = patch_center.get_array()
        patch_center_value = patch_center_value[patch_center_value != 0]
        #Calculate the average curvedness per triangle for each patch
        for i in patch_center_value:
            # Extract the curvature of the triangles which have patch_number = i
            curvedness = tg.graph.vp.curvedness_VV.get_array()
            #print(f"curvedness: {curvedness}")
            #fetch the vertex property patch_number
            patch_number = tg.graph.vp.patch_number
            curvedness_patch = curvedness[patch_number.a == i]
            print(f"number of triangles in patch {i} is: {len(curvedness_patch)}")
            #create a mask to filter out NaNs
            mask_patch = ~np.isnan(curvedness_patch)
            curvedness_patch = curvedness_patch[mask_patch]
            #create a mask to filter out 0
            mask_patch = curvedness_patch != 0
            curvedness_patch = curvedness_patch[mask_patch]
            print(f"number of triangles in patch {i} after filtering out NaNs and 0 is: {len(curvedness_patch)}")
            #calculate the average curvedness of the patch per triangle
            average_curvedness_patch_per_triangle = np.sum(curvedness_patch) / len(curvedness_patch)
            print(f"average curvedness of patch {i} per triangle is: {average_curvedness_patch_per_triangle}")

            # Extract the curvedness of the triangle whivh have patch_random_number = i
            patch_random_number = tg.graph.vp.patch_random_number
            curvedness_random = curvedness[patch_random_number.a == i]
            print(f"number of triangles in random patch {i} is: {len(curvedness_random)}")
            #create a mask to filter out NaNs
            mask_random = ~np.isnan(curvedness_random)
            curvedness_random = curvedness_random[mask_random]
            #create a mask to filter out 0
            mask_random = curvedness_random != 0
            curvedness_random = curvedness_random[mask_random]
            print(f"number of triangles in random patch {i} after filtering out NaNs and 0 is: {len(curvedness_random)}")
            #calculate the average curvedness of the random patch per triangle
            average_curvedness_random_patch_per_triangle = np.sum(curvedness_random) / len(curvedness_random)
            patch_number = i
            patch_random_number = i
            print(f"average curvedness of random patch {i} per triangle is: {average_curvedness_random_patch_per_triangle}")
            # Append the average _curvedness of the patch, average curvedness of the random patch, patch number, and random patch number to the list  
            average_curvedness_data.append({
                "tomogram": tomogram_name,
                "patch_number": patch_number,
                "average_curvedness_patch_per_triangle": average_curvedness_patch_per_triangle,
                "patch_random_number": patch_random_number,
                "average_curvedness_random_patch_per_triangle": average_curvedness_random_patch_per_triangle

            })
    # Create a dataframe for the collected data
    df = pd.DataFrame(average_curvedness_data)
    # Save the dataframe to a CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    average_curvedness_calculation()
