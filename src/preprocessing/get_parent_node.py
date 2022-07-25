from src.utils import logger
from src.features.get_svg_size_pos import *


def get_clip_paths(file):
    """ Identify clip paths in an SVG.

    Args:
        file (str): Path of SVG.

    Returns:
        list: Animation IDs of all paths that have a clip-path as a parent node.

    """
    doc = minidom.parse(file)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    clip_paths=[]
    for i in range(len(elements)):
        if elements[i].parentNode.nodeName == "clipPath":
            clip_paths.append(elements[i].attributes['animation_id'].value)
    return clip_paths


def get_background_paths(file):
    """ Identify background by checking if its bbox size is nearly as big as the complete SVG.

    Args:
        file (str): Path of SVG.

    Returns:
        list: Animation IDs of all background candidates.

    """
    doc = minidom.parse(file)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    background_paths = []
    width, height = get_svg_size(file)
    surface_svg = width * height
    for i in range(len(elements)):
        try:
            xmin, xmax, ymin, ymax = get_path_bbox(file,i)
            surface_path = (xmax-xmin)*(ymax-ymin)
            if surface_path > (0.98*surface_svg):
                background_paths.append(elements[i].attributes['animation_id'].value)
        except Exception as e:
            logger.error(f'Could not retrieve bbox for animation ID {i} in file {file}: {e}')
    return background_paths
