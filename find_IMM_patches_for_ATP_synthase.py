import click
import numpy as np
import starfile
from scipy.spatial import cKDTree
from pycurv import TriangleGraph, io
from graph_tool import load_graph
import random

np.bool = bool

@click.command()
@click.option('--input_star', type=click.Path(exists=True), help='Path to the input star file')
@click.option('--imm_graph', type=click.Path(exists=True), help='Path to the IMM graph (.gt) file')
@click.option('--output_folder', type=click.Path(), help='Path to the output folder')

def find_IMM_patches_for_ATP_synthase(input_star, imm_graph, output_folder):
    # read the star file with ATP syntahse particle coordinates
    star = starfile.read(input_star)

    # divide the pixel size column by 10 to convert the pixel size to nanometer
    star['rlnPixelSize'] = star['rlnPixelSize'] / 10
    # multiply the coordinate with pixel size to get the real coordinates in the unit of nanometer
    star['rlnCoordinateX'] = star['rlnCoordinateX'] * star['rlnPixelSize']
    star['rlnCoordinateY'] = star['rlnCoordinateY'] * star['rlnPixelSize']
    star['rlnCoordinateZ'] = star['rlnCoordinateZ'] * star['rlnPixelSize']
    # convert the coordinates to a numpy array
    a = np.array(star[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']])
    
    # load the IMM graph
    tg1 = TriangleGraph()
    tg1.graph = load_graph(imm_graph)
    xyz1 = tg1.graph.vp.xyz.get_2d_array([0,1,2]).transpose()

    # find the nearest IMM triangle to each ATP synthase particle
    tree1 = cKDTree(xyz1)
    min_d_1, min_i_1 = tree1.query(a, k=1)

    # filter min_i_1 by min_d_1 to remove the synthase from the other mitochondria
    min_i_1 = min_i_1[min_d_1 <= 24]

    # extract the coordinates of the nearest IMM triangle by using the indices of the nearest IMM triangle
    IMM_center_Index = min_i_1
    #print("IMM_center_Index:", IMM_center_Index)
    i = len(IMM_center_Index)
    print("the number of patch:", i)

    # generate a sequence of numbers from 0 to i with a step of 1
    sequence = np.arange(0, i, 1)
    print(sequence)

    # add patch_center as a new vertex property to the IMM graph
    patch_center = tg1.graph.new_vertex_property("int")

    # Assign the sequence values to the corresponding vertices in the IMM graph
    for idx, vertex_index in enumerate(IMM_center_Index):
        # The vertex_index refers to the actual vertex in the IMM graph
        vertex = tg1.graph.vertex(vertex_index)
        patch_center[vertex] = sequence[idx] + 1  # Add 1 to match your desired output

    # Save the patch_center property back to the graph
    tg1.graph.vertex_properties['patch_center'] = patch_center

    #create a new vertex property for the IMM graph
    patch_number = tg1.graph.new_vertex_property("int")

    for number in sequence + 1:
        # extract the coordinates of the IMM triangle with the patch_center number
        IMM_center_coordinates = xyz1[patch_center.a == number]
    
        # Check if IMM_center_coordinates is empty
        if IMM_center_coordinates.shape[0] == 0:
            print(f"No IMM center coordinates found for patch {number}")
            continue  # Skip to the next patch
    
        #print(IMM_center_coordinates)
        
        # Find the IMM triangles that are within 12 nm of the IMM triangle with the patch_center number
        distances = np.linalg.norm(xyz1 - IMM_center_coordinates, axis=1)
        indices_within_range = np.where(distances <= 12)
    
        # Write the number of the patch to the patch_number property according to the indices within range
        for idx, vertex_index in enumerate(indices_within_range[0]):
            vertex = tg1.graph.vertex(vertex_index)
            patch_number[vertex] = number

    # Save the patch_number property back to the graph
    tg1.graph.vertex_properties['patch_number'] = patch_number

    #convert xyz1 to a list of tuples
    xyz1_tuples = [tuple(coord) for coord in xyz1]
    #print("xyz1_tuples:", xyz1_tuples)

    #generate random IMM coordinates
    random_coordinates = generate_random_coordinates(xyz1_tuples, i, min_distance=12)
    #print(random_coordinates)
    #find the indicies of the random coordinates in the xyz1
    random_coordinates_indices = [xyz1_tuples.index(coord) for coord in random_coordinates]
    print(' the number of random coordinates:', len(random_coordinates_indices))
    # add patch_random_center as a new vertex property to the IMM graph
    patch_random_center = tg1.graph.new_vertex_property("int")
    # Assign the sequence values to the corresponding vertices in the IMM graph according to the random coordinates indicies
    for idx, vertex_index in enumerate(random_coordinates_indices):
        # The vertex_index refers to the actual vertex in the IMM graph
        vertex = tg1.graph.vertex(vertex_index)
        patch_random_center[vertex] = sequence[idx] + 1
    # Save the patch_random_center property back to the graph
    tg1.graph.vertex_properties['patch_random_center'] = patch_random_center
    
    #create a new vertex property for the IMM graph
    patch_random_number = tg1.graph.new_vertex_property("int")
    for number in sequence + 1:
        # extract the coordinaes of the IMM triangle with the patch_random_center number
        random_IMM_center_coordinates = xyz1[patch_random_center.a == number]
        #print(random_IMM_center_coordinates)
        #find the IMM triangles which are within 12 nm of the IMM triangle with the patch_random_center number
        distances = np.linalg.norm(xyz1 - random_IMM_center_coordinates, axis=1)
        indices_within_range = np.where(distances <= 12)
        #write the number of the patch to the patch_random_number property according to the indices within range
        for idx, vertex_index in enumerate(indices_within_range[0]):
            vertex = tg1.graph.vertex(vertex_index)
            patch_random_number[vertex] = number
    # Save the patch_random_number property back to the graph   
    tg1.graph.vertex_properties['patch_random_number'] = patch_random_number


    # Save the IMM graph with the patch_center property to an output folder
    basename = imm_graph.split("/")[-1]
    surface1 = output_folder + "/" + basename[:-3] + "_individual_patch.vtp"
    new_imm_graph = output_folder + "/" + basename[:-3] + "_individual_patch.gt"

    # Optionally, save the modified graph to a file
    tg1.graph.save(new_imm_graph)

    # Optionally, save the modified graph to a VTP file
    surf1 = tg1.graph_to_triangle_poly()
    io.save_vtp(surf1, surface1)
    print("Patch saved as a vertex property in the IMM graph")

def generate_random_coordinates(set_of_coordinates, num_coordinates, min_distance=12):
    if len(set_of_coordinates) < num_coordinates:
        raise ValueError("The provided set of coordinates is too small to select the required number of random coordinates.")

    random_coordinates = []
    remaining_coordinates = set_of_coordinates.copy()

    while len(random_coordinates) < num_coordinates and remaining_coordinates:
        candidate = random.choice(remaining_coordinates)
        if all(calculate_distance(candidate, existing_coord) >= min_distance for existing_coord in random_coordinates):
            random_coordinates.append(candidate)
            remaining_coordinates.remove(candidate)
    
    if len(random_coordinates) < num_coordinates:
        raise ValueError("It is not possible to select the required number of coordinates with the given minimum distance constraint.")
    
    return random_coordinates

def calculate_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

if __name__ == '__main__':
    find_IMM_patches_for_ATP_synthase()
        




