#!/usr/bin/env python3
"""Distance from a membrane surface graph to a fibril segmentation mask.

For every vertex of one or more pycurv/graph-tool TriangleGraph surfaces
(``.gt`` files), find the nearest voxel of a fibril segmentation mask (an
MRC volume) and record that distance as a new vertex property (named
``mrc_distance`` by default; see ``--column-name``). Results are written as
an annotated ``.gt`` graph, a ``.vtp`` surface (for viewing in ParaView), and
-- when a matching ``<graph>.csv`` morphometrics table exists next to the
graph file -- an updated CSV with the same column added.
"""
import glob
import os

import click
import matplotlib
import numpy as np
import pandas as pd
from graph_tool import load_graph
from pycurv import TriangleGraph, io
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_fibril_mask_coords(mask_path, apix, min_label):
    """Load an MRC mask and return physical-unit (x, y, z) coords of fibril voxels."""
    import mrcfile

    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = mrc.data
    if np.min(mask) < 0:
        mask = mask - np.min(mask)

    labels, counts = np.unique(mask, return_counts=True)
    click.echo("Voxel counts per label in mask:")
    for label, count in zip(labels, counts):
        click.echo(f"  label {label:g}: {count} voxels")

    # MRC arrays are indexed (z, y, x); reorder to (x, y, z) and scale to physical units.
    coords_zyx = np.array(np.where(mask >= min_label)).T
    coords_xyz = coords_zyx[:, [2, 1, 0]] * apix
    click.echo(f"Fibril mask: {len(coords_xyz)} voxels with label >= {min_label}")
    if len(coords_xyz) == 0:
        raise click.ClickException(
            f"No voxels in '{mask_path}' have label >= {min_label}; "
            "check --min-label and the mask file."
        )
    click.echo(f"  coord min: {coords_xyz.min(axis=0)}")
    click.echo(f"  coord max: {coords_xyz.max(axis=0)}")
    return coords_xyz


def process_membrane_graph(graph_path, fibril_tree, output_dir, unit, make_plots, n_check, rng, column_name):
    """Annotate one membrane graph with distances to the fibril mask and save outputs."""
    click.echo(f"\n=== {graph_path} ===")
    tg = TriangleGraph()
    tg.graph = load_graph(graph_path)

    if column_name in tg.graph.vertex_properties:
        raise click.ClickException(
            f"Graph '{graph_path}' already has a vertex property named '{column_name}'. "
            "Pick a different --column-name."
        )

    xyz = tg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()
    click.echo(f"  membrane vertices: {len(xyz)}")

    distances, nearest_idx = fibril_tree.query(xyz, k=1)
    click.echo(
        f"  distance to fibril ({unit}): min={distances.min():.2f} "
        f"max={distances.max():.2f} mean={distances.mean():.2f}"
    )

    _, dup_counts = np.unique(nearest_idx, return_counts=True)
    n_shared = int(np.sum(dup_counts > 1))
    if n_shared:
        click.echo(
            f"  note: {n_shared} fibril voxel(s) are the nearest match for more than "
            "one membrane vertex (expected -- the fibril mask is usually coarser than the mesh)"
        )

    stem = os.path.splitext(os.path.basename(graph_path))[0]

    if make_plots:
        plt.figure()
        plt.scatter(fibril_tree.data[:, 0], fibril_tree.data[:, 1], s=1, c="tab:blue", label="fibril mask")
        plt.scatter(xyz[:, 0], xyz[:, 1], s=1, c="tab:red", label="membrane")
        plt.xlabel(f"x ({unit})")
        plt.ylabel(f"y ({unit})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{stem}_mask_vs_membrane.png"))
        plt.close()

        plt.figure()
        plt.hist(distances, bins=100)
        plt.xlabel(f"Distance to nearest fibril ({unit})")
        plt.ylabel("Number of membrane vertices")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{stem}_mrc_distance_histogram.png"))
        plt.close()

    # Vertex indices from get_2d_array/xyz run 0..N-1 in the same order as the
    # property map's underlying array, so this assigns each vertex its own distance.
    mrc_distance = tg.graph.new_vertex_property("float")
    mrc_distance.a = distances
    tg.graph.vertex_properties[column_name] = mrc_distance

    out_gt = os.path.join(output_dir, f"{stem}_mrc_distance.gt")
    out_vtp = os.path.join(output_dir, f"{stem}_mrc_distance.vtp")
    tg.graph.save(out_gt)
    io.save_vtp(tg.graph_to_triangle_poly(), out_vtp)
    click.echo(f"  saved {out_gt}")
    click.echo(f"  saved {out_vtp}")

    csv_path = os.path.splitext(graph_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        click.echo(f"  WARNING: no matching CSV at '{csv_path}', skipping CSV export", err=True)
        return

    df = pd.read_csv(csv_path)
    if column_name in df.columns:
        raise click.ClickException(
            f"CSV '{csv_path}' already has a column named '{column_name}'. "
            "Pick a different --column-name."
        )
    if len(df) != len(xyz):
        click.echo(
            f"  WARNING: CSV has {len(df)} rows but graph has {len(xyz)} vertices; "
            "row order may not match vertex order",
            err=True,
        )

    df[column_name] = mrc_distance.a
    out_csv = os.path.join(output_dir, os.path.basename(csv_path).replace(".csv", "_with_mrc_dist.csv"))
    df.to_csv(out_csv, index=False)
    click.echo(f"  saved {out_csv}")

    if n_check > 0 and len(xyz) >= n_check:
        check_idx = np.sort(rng.choice(len(xyz), size=n_check, replace=False))
        mismatches = [
            idx
            for idx in check_idx
            if not np.isclose(
                mrc_distance[tg.graph.vertex(int(idx))], df.loc[idx, column_name], rtol=1e-9
            )
        ]
        if mismatches:
            click.echo(
                f"  SANITY CHECK FAILED: {len(mismatches)}/{n_check} spot-checked vertices "
                f"disagree between the graph and the CSV (indices: {mismatches})",
                err=True,
            )
        else:
            click.echo(f"  sanity check passed: {n_check} spot-checked vertices agree")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--mask-mrc",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="MRC volume containing the fibril segmentation (binary or multi-label mask).",
)
@click.option(
    "--graph-glob",
    "-g",
    required=True,
    help="Glob pattern matching one or more membrane surface .gt graphs "
    "(pycurv/graph-tool TriangleGraph), e.g. \"1_morph/*_refined.gt\". "
    "Quote it so your shell doesn't expand it first.",
)
@click.option(
    "--output-dir",
    "-o",
    default="mrc_distance_output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory for annotated graphs (.gt/.vtp), updated CSVs, and diagnostic plots.",
)
@click.option(
    "--apix",
    type=float,
    required=True,
    help="Physical size of one mask voxel edge, in the unit given by --unit "
    "(e.g. the tomogram pixel size after binning). Must be in the same unit as the "
    "membrane graph's own coordinates, or distances will be wrong.",
)
@click.option(
    "--unit",
    default="nm",
    show_default=True,
    help="Unit label used on plots and in printed summaries. Cosmetic only -- make sure "
    "it actually matches --apix.",
)
@click.option(
    "--min-label",
    type=int,
    default=1,
    show_default=True,
    help="Minimum mask voxel value counted as fibril (voxels with value >= this are "
    "included). Use 1 for a plain binary 0/1 mask; raise it to select one class out of "
    "a multi-label segmentation.",
)
@click.option(
    "--n-check",
    type=int,
    default=25,
    show_default=True,
    help="Number of vertices to spot-check for graph/CSV consistency after writing the "
    "CSV. Set to 0 to skip.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for the spot-check sample (default: a fresh random sample each run).",
)
@click.option(
    "--plots/--no-plots",
    default=True,
    help="Write per-graph diagnostic PNGs (mask-vs-membrane scatter + distance histogram).",
)
@click.option(
    "--column-name",
    default="mrc_distance",
    show_default=True,
    help="Name for the new vertex property (in the .gt/.vtp) and the new CSV column. "
    "If a graph or CSV already has a property/column with this name, the run stops "
    "with an error so you can rerun with a different --column-name.",
)
def main(mask_mrc, graph_glob, output_dir, apix, unit, min_label, n_check, seed, plots, column_name):
    """Annotate membrane graph(s) with distance-to-fibril, per vertex.

    \b
    Example:
      python surface_distance_from_mrc.py \\
          --mask-mrc 208_fibers_binary_bin8.mrc \\
          --graph-glob "1_morph/208_bin8_manual_clean_merged_segmented_lyso?.*_refined.gt" \\
          --apix 1.741 --output-dir lyso_fibril_patches

    For each matched graph, this looks for a CSV with the same name (extension
    swapped to .csv) in the same folder -- typically a pycurv morphometrics table --
    and, if found, writes a copy of it with the new distance column added.
    """
    graph_paths = sorted(glob.glob(graph_glob))
    if not graph_paths:
        raise click.ClickException(f"No graph files matched --graph-glob '{graph_glob}'")
    click.echo(f"Found {len(graph_paths)} membrane graph(s) matching '{graph_glob}'")

    os.makedirs(output_dir, exist_ok=True)

    fibril_coords = load_fibril_mask_coords(mask_mrc, apix, min_label)
    fibril_tree = cKDTree(fibril_coords)
    rng = np.random.default_rng(seed)

    for graph_path in graph_paths:
        process_membrane_graph(graph_path, fibril_tree, output_dir, unit, plots, n_check, rng, column_name)


if __name__ == "__main__":
    main()
