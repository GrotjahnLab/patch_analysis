import os
import glob
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pycurv import TriangleGraph, io
from graph_tool import load_graph, GraphView
from intradistance_verticality import export_csv

np.bool = bool

@click.command()
@click.option('--gt_folder', required=True, help='Path to the folder containing .gt files')
@click.option('--output_folder', required=True, help='Path to the folder where the output files will be saved')
def extract_single_patch(gt_folder, output_folder):
    """
    Process all .gt files in a folder by detecting the unique patch_number values
    in each file, applying a filter based on patch_number, and generating .gt,
    .vtp, and .csv files for each patch.
    """
    # Find all .gt files in the gt_folder
    gt_files = glob.glob(os.path.join(gt_folder, '*.gt'))

    for gt_file in gt_files:
        print(f"Processing file: {gt_file}")

        # Load the graph
        tg = TriangleGraph()
        tg.graph = load_graph(gt_file)

        # Fetch the vertex property 'patch_number'
        patch_number = tg.graph.vp.patch_number  # This is a VertexPropertyMap

        # Extract unique patch numbers from the graph
        unique_patch_numbers = set(patch_number[v] for v in tg.graph.vertices())
        # remove 0 from the set
        unique_patch_numbers.discard(0)
        print(f"Unique patch numbers found: {unique_patch_numbers}")

        # Extract the base name of the file (without extension) for output file names
        base_name = os.path.splitext(os.path.basename(gt_file))[0]

        # Loop through the unique patch numbers
        for patch_num in unique_patch_numbers:
            #load gt_file
            tg.graph = load_graph(gt_file)
            # Create a vertex filter based on patch_number
            vertex_filter = tg.graph.new_vertex_property("bool")
            for v in tg.graph.vertices():
                vertex_filter[v] = patch_number[v] == patch_num

            # Apply the vertex filter
            tg.graph.set_vertex_filter(vertex_filter)

            print(f"Filtered vertices for patch {patch_num}:", tg.graph.num_vertices())

            # Create a GraphView to keep only the filtered vertices
            filtered_graph = GraphView(tg.graph, vfilt=vertex_filter)

            # Remove unfiltered vertices and save the graph
            filtered_graph.purge_vertices()

            # Define file paths for saving the outputs to output_folder
            output_base = os.path.join(output_folder, f'{base_name}_patch{patch_num}')

            # Save the filtered graph to a new .gt file
            filtered_graph.save(f'{output_base}.gt')

            # Load the filtered graph and save as .vtp
            tg1 = TriangleGraph()
            tg1.graph = load_graph(f'{output_base}.gt')
            surf = tg1.graph_to_triangle_poly()
            io.save_vtp(surf, f'{output_base}.vtp')

            # Export the CSV file
            export_csv(tg1, f'{output_base}.csv')

if __name__ == '__main__':
    extract_single_patch()
