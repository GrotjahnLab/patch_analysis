import click
import pandas as pd
import numpy as np
import mrcfile
import scipy.interpolate as interp
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial import cKDTree
import starfile as sf

# Constants
workfolder = '/scratch1/users/atty/ATP_patch_analysis/new_ATP_synthase_patches/extracted_patches_untreated/'
components = ["IMM"]

# Load data from a CSV file using pandas and extract the xyz coordinates and n_v normal values
def load_csv(filename, voxsize):
    df = pd.read_csv(filename)
    x = np.array(df['xyz_x'])/voxsize
    y = np.array(df['xyz_y'])/voxsize
    z = np.array(df['xyz_z'])/voxsize
    xyz = np.array([x,y,z])
    n_v = np.array([df['n_v_x']/voxsize,df['n_v_y']/voxsize,df['n_v_z']/voxsize])
    center_idx = np.where(df['patch_center'] > 0)
    print(f"patch center index: {center_idx}")

    # Handle empty center index and multiple center indices
    if len(center_idx[0]) > 1:
        print(f"Multiple patch centers found in {filename}. Skipping.")
        return xyz, n_v, None
    if len(center_idx[0]) == 0:
        print(f"No patch center found in {filename}. Skipping.")
        return xyz, n_v, None

    center_xyz = np.squeeze(xyz[:, center_idx])
    print(f"center_xyz: {center_xyz}")
    ##only keep the coordinates which are within 10nm/voxsize of the center
    #patch_size = 10 / voxsize
    #print(f"number of triangles before filtering: {len(xyz[0])}")
    #xyz = xyz[:, np.where(np.linalg.norm(xyz - center_xyz[:, None], axis=0) <= patch_size)]
    #print(f"number of triangles after filtering: {len(xyz[0])}")
    return xyz, n_v, center_xyz

def check_normal_vector(starfile, n_v, center_xyz, voxsize):
    star = sf.read(starfile)
    star['rlnDetectorPixelSize'] = star['rlnDetectorPixelSize'] / 10 / voxsize
    star['rlnCoordinateX'] *= star['rlnDetectorPixelSize']
    star['rlnCoordinateY'] *= star['rlnDetectorPixelSize']
    star['rlnCoordinateZ'] *= star['rlnDetectorPixelSize']
    a = np.array(star[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']])
    tree = cKDTree(a)
    min_d, min_i = tree.query(center_xyz, k=1)
    ATP_synthase_center = a[min_i]
    print(f"ATP_synthase_center: {ATP_synthase_center}")
    vector = ATP_synthase_center - center_xyz
    #print(f"vector: {vector}")
    #print(f"n_v: {n_v}")
    n_v_flipped = np.zeros_like(n_v)
    for i in range(len(n_v[0])):
        if np.dot(n_v[:, i], vector) < 0:
            n_v_flipped[:, i] = -n_v[:, i]
        else:
            n_v_flipped[:, i] = n_v[:, i]
    #print(f"n_v_flipped: {n_v_flipped}")
    return n_v_flipped

def load_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        print(mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z)
        data = mrc.data
        data = np.swapaxes(data, 0, 2)
        voxsize = mrc.voxel_size.x / 10
        print(f"Voxel size: {voxsize}")
        print(data.shape)
        data_matrix = (np.arange(data.shape[0]), np.arange(data.shape[1]), np.arange(data.shape[2]))
    return data, data_matrix, voxsize

def interpolate(data, data_matrix, xyz, n_v_flipped, nsamples=161):
    averages = []
    nm = np.linspace(-10, 30, nsamples)
    #print(nm)
    value_array = np.empty((len(n_v_flipped[0]), len(nm)))
    for i in range(len(n_v_flipped[0])):
        locindices = [xyz[:, i] + j * n_v_flipped[:, i] for j in nm]
        value_array_temp = interp.interpn(data_matrix, data, locindices, method="linear", bounds_error=False, fill_value=None)
        averages.append(value_array_temp[int((nsamples + 1) / 2)])
        value_array[i] = value_array_temp
    print(np.mean(averages))
    return value_array

def run_mrc(mrcfile, starfile, csv_file_header):
    components = ["IMM"]
    files = glob(workfolder + f"{csv_file_header}.AVV_rh8_refined_individual_patch_patch*.csv")
    print(f'csv_files: {len(files)}')
    print(f'starfile: {starfile}')
    data, data_matrix, voxsize = load_mrc(mrcfile)
    print(data.shape)
    for file in files:
        print(file)
        xyz, n_v, center_xyz = load_csv(file, voxsize)
        # Skip file if center_xyz is None
        if center_xyz is None:
            continue
        n_v_flipped = check_normal_vector(starfile, n_v, center_xyz, voxsize)
        value_array = interpolate(data, data_matrix, xyz, n_v_flipped)
        print(file[:-4] + f"_sampling.csv")
        np.savetxt(file[:-4] + f"_sampling.csv", value_array, delimiter=",")

@click.command()
@click.option('--mrcfile', help='Path to the MRC file.')
@click.option('--starfile', help='Path to the STAR file.')
@click.option('--csv_file_header', help='Header of the CSV files (before .AVV).')
def main(mrcfile, starfile, csv_file_header):
    run_mrc(mrcfile, starfile, csv_file_header)

if __name__ == "__main__":
    main()
