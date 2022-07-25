import os
import glob
from concurrent import futures
from tqdm import tqdm
import pandas as pd
from src.preprocessing.deepsvg.svglib.svg import SVG


def get_svg_meta_data(data_folder="data/svgs"):
    """ Get meta data of all SVGs in a given folder.

    Note: There are some elements (like text tags or matrices or clip paths) that can't be processed here. The meta
    file only considers "normal" elements.

    Args:
        data_folder (str): Path of the folder containing all SVGs.

    Returns:
        pd.DataFrame: Dataframe containing metadata of SVGs.

    """
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        svg_files = glob.glob(os.path.join(data_folder, "*.svg"))
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [
                executor.submit(_get_svg_meta_data, svg_file, meta_data)
                for svg_file in svg_files]
            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)
    df = pd.DataFrame(meta_data.values())
    return df


def _get_svg_meta_data(svg_file, meta_data):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    #svg = SVG.load_svg(svg_file)  # THIS ONE
    # svg.fill_(False)
    # svg.normalize()
    # svg.zoom(0.9)
    # svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])

    #svg.canonicalize(normalize=True)  # THIS ONE

    svg = _canonicalize(svg_file, normalize=True)

    # svg = svg.simplify_heuristic()

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]
    start_pos = [path_group.svg_paths[0].start_pos for path_group in svg.svg_path_groups]

    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups),
        "start_pos": start_pos
    }


def _canonicalize(svg_file, normalize=False):
    svg = SVG.load_svg(svg_file)
    svg.to_path().simplify_arcs()

    if normalize:
        svg.normalize()

    #svg.split_paths()
    svg.filter_consecutives()
    svg.filter_empty()
    svg._apply_to_paths("reorder")
    svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
    svg._apply_to_paths("canonicalize")
    svg.recompute_origins()

    svg.drop_z()

    return svg


def _apply_to_paths_of_svg(svg, method, *args, **kwargs):
    for path_group in svg.svg_path_groups:
        getattr(path_group, method)(*args, **kwargs)
    return svg
