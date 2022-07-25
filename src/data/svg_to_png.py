import os
from cairosvg import svg2png


def convert_svgs_in_folder(folder):
    """ Convert all SVGs in a given folder to PNG. SVGs are deleted after PNGs have been created.

    Args:
        folder (str): Path of folder containing all SVG files.

    Returns:
        list: List of converted files.

    """
    paths_list = []
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            file = folder + '/' + file
            convert_svg(file)
            # create path list
            paths_list.append(file.replace('.svg', '.png'))
            os.remove(file)
    return paths_list


def convert_svg(file):
    """ Convert one SVG to PNG. Requires Cairosvg.

    Args:
        file (str): Path of SVG that needs to be converted.

    """
    # Change name and path for writing element pngs
    filename = file.replace('.svg', '')
    # Convert svg to png
    svg2png(url=file, write_to=filename + '.png')
