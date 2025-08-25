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
def average_thickness_calculation(gt_dir, output_csv):
    # Create an empty list to store the average thickness data
    average_thickness_data = []

    # Collect all .gt files in the specified directory
    gt_graphs = glob.glob(os.path.join(gt_dir, "*.gt"))

    #count the average thickness of each patch for each tomogram
    for graph in gt_graphs:
        # Load the graph
        tg = TriangleGraph()
        tg.graph = load_graph(graph)
        # Extract the tomogram name before .labels_IMM.AVV_rh8_refined_with_patches.gt
        tomogram_name = os.path.basename(graph).split('.labels_IMM.AVV_rh8_refined_individual_patch.gt')[0]
        print(tomogram_name)

        # Fetch the vertex property patch_center and patch_random_center
        patch_center = tg.graph.vp.patch_center
        patch_random_center = tg.graph.vp.patch_random_center
        # find the value of patch center that is not 0
        patch_center_value = patch_center.get_array()
        patch_center_value = patch_center_value[patch_center_value != 0]
        #Calculate the average thickness per triangle for each patch
        for i in patch_center_value:
            # Extract the thiiickness of the triangles which have patch_number = i
            thickness = tg.graph.vp.thickness.get_array()
            #fetch the vertex property patch_number
            patch_number = tg.graph.vp.patch_number
            thickness_patch = thickness[patch_number.a == i]
            print(f"number of triangles in patch {i} is: {len(thickness_patch)}")
            #create a mask to filter out NaNs
            mask_patch = ~np.isnan(thickness_patch)
            thickness_patch = thickness_patch[mask_patch]
            #create a mask to filter out 0
            mask_patch = thickness_patch != 0
            thickness_patch = thickness_patch[mask_patch]
            print(f"number of triangles in patch {i} after filtering out NaNs and 0 is: {len(thickness_patch)}")
            #calculate the average thickness of the patch per triangle
            average_thickness_patch_per_triangle = np.sum(thickness_patch) / len(thickness_patch)
            print(f"average thickness of patch {i} per triangle is: {average_thickness_patch_per_triangle}")

            # Extract the thickness of the triangle whivh have patch_random_number = i
            patch_random_number = tg.graph.vp.patch_random_number
            thickness_random = thickness[patch_random_number.a == i]
            print(f"number of triangles in random patch {i} is: {len(thickness_random)}")
            #create a mask to filter out NaNs
            mask_random = ~np.isnan(thickness_random)
            thickness_random = thickness_random[mask_random]
            #create a mask to filter out 0
            mask_random = thickness_random != 0
            thickness_random = thickness_random[mask_random]
            print(f"number of triangles in random patch {i} after filtering out NaNs and 0 is: {len(thickness_random)}")
            #calculate the average thickness of the random patch per triangle
            average_thickness_random_patch_per_triangle = np.sum(thickness_random) / len(thickness_random)
            patch_number = i
            patch_random_number = i
            print(f"average thickness of random patch {i} per triangle is: {average_thickness_random_patch_per_triangle}")
            # Append the average _thickness of the patch, average thickness of the random patch, patch number, and random patch number to the list  
            average_thickness_data.append({
                "tomogram": tomogram_name,
                "patch_number": patch_number,
                "average_thickness_patch_per_triangle": average_thickness_patch_per_triangle,
                "patch_random_number": patch_random_number,
                "average_thickness_random_patch_per_triangle": average_thickness_random_patch_per_triangle

            })
    # Create a dataframe for the collected data
    df = pd.DataFrame(average_thickness_data)
    # Save the dataframe to a CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    average_thickness_calculation()
