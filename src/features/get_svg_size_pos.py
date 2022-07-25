from xml.dom import minidom
from svgpathtools import svg2paths


def get_svg_size(file):
    """ Get width and height of an SVG.

    Args:
        file (str): Path of SVG file.

    Returns:
        float, float: Width and height of SVG.

    """
    doc = minidom.parse(file)
    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')

    if width != "" and height != "":
        if not width[-1].isdigit():
            width = width.replace('px', '').replace('pt', '')
        if not height[-1].isdigit():
            height = height.replace('px', '').replace('pt', '')

    if width == "" or height == "" or not width[-1].isdigit() or not height[-1].isdigit():
        # get bounding box of svg
        xmin_svg, xmax_svg, ymin_svg, ymax_svg = 100, -100, 100, -100
        paths, _ = svg2paths(file)
        for path in paths:
            xmin, xmax, ymin, ymax = path.bbox()
            if xmin < xmin_svg:
                xmin_svg = xmin
            if xmax > xmax_svg:
                xmax_svg = xmax
            if ymin < ymin_svg:
                ymin_svg = ymin
            if ymax > ymax_svg:
                ymax_svg = ymax
            width = xmax_svg - xmin_svg
            height = ymax_svg - ymin_svg

    return float(width), float(height)


def get_svg_bbox(file):
    """ Get bounding box coordinates of an SVG.

    xmin, ymin: Upper left corner.

    xmax, ymax: Lower right corner.

    Args:
        file (str): Path of SVG file.

    Returns:
         float, float, float, float: Bounding box of SVG (xmin, xmax, ymin, ymax).

    """
    try:
        paths, _ = svg2paths(file)
    except Exception as e:
        print(f"{file}: svg2path fails. SVG bbox is computed by using get_svg_size. {e}")
        width, height = get_svg_size(file)
        return 0, width, 0, height

    xmin_svg, xmax_svg, ymin_svg, ymax_svg = 100, -100, 100, -100
    for path in paths:
        try:
            xmin, xmax, ymin, ymax = path.bbox()
            if xmin < xmin_svg:
                xmin_svg = xmin
            if xmax > xmax_svg:
                xmax_svg = xmax
            if ymin < ymin_svg:
                ymin_svg = ymin
            if ymax > ymax_svg:
                ymax_svg = ymax
        except:
            pass

    return xmin_svg, xmax_svg, ymin_svg, ymax_svg


def get_path_bbox(file, animation_id):
    """ Get bounding box coordinates of a path in an SVG.

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element.

    Returns:
        float, float, float, float: Bounding box of path (xmin, xmax, ymin, ymax).

    """
    try:
        paths, attributes = svg2paths(file)
    except Exception as e1:
        print(f"{file}, animation ID {animation_id}: svg2path fails and path bbox cannot be computed. {e1}")
        return 0, 0, 0, 0

    for i, path in enumerate(paths):
        if attributes[i]["animation_id"] == str(animation_id):
            try:
                xmin, xmax, ymin, ymax = path.bbox()
                return xmin, xmax, ymin, ymax
            except Exception as e2:
                print(f"{file}, animation ID {animation_id}: svg2path fails and path bbox cannot be computed. {e2}")
                return 0, 0, 0, 0


def get_midpoint_of_path_bbox(file, animation_id):
    """ Get midpoint of bounding box of path.

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element.

    Returns:
        float, float: Midpoint of bounding box of path (x_midpoint, y_midpoint).

    """
    try:
        xmin, xmax, ymin, ymax = get_path_bbox(file, animation_id)
        x_midpoint = (xmin + xmax) / 2
        y_midpoint = (ymin + ymax) / 2

        return x_midpoint, y_midpoint
    except Exception as e:
        print(f'Could not get midpoint for file {file} and animation ID {animation_id}: {e}')
        return 0, 0


def get_bbox_of_multiple_paths(file, animation_ids):
    """ Get bounding box of multiple paths in an SVG.

    Args:
        file (str): Path of SVG file.
        animation_ids (list(int)): List of element IDs.

    Returns:
        float, float, float, float: Bounding box of given paths (xmin, xmax, ymin, ymax).

    """
    try:
        paths, attributes = svg2paths(file)
    except Exception as e1:
        print(f"{file}: svg2path fails and bbox of multiple paths cannot be computed. {e1}")
        return 0, 0, 0, 0

    xmin_paths, xmax_paths, ymin_paths, ymax_paths = 100, -100, 100, -100

    for i, path in enumerate(paths):
        if attributes[i]["animation_id"] in list(map(str, animation_ids)):
            try:
                xmin, xmax, ymin, ymax = path.bbox()
                if xmin < xmin_paths:
                    xmin_paths = xmin
                if xmax > xmax_paths:
                    xmax_paths = xmax
                if ymin < ymin_paths:
                    ymin_paths = ymin
                if ymax > ymax_paths:
                    ymax_paths = ymax
            except:
                pass

    return xmin_paths, xmax_paths, ymin_paths, ymax_paths


def get_relative_path_pos(file, animation_id):
    """ Get relative position of a path in an SVG.

    Args:
        file (string): Path of SVG file.
        animation_id (int): ID of element.

    Returns:
        float, float: Relative x- and y-position of path.

    """
    path_midpoint_x, path_midpoint_y = get_midpoint_of_path_bbox(file, animation_id)
    svg_xmin, svg_xmax, svg_ymin, svg_ymax = get_svg_bbox(file)
    rel_x_position = (path_midpoint_x - svg_xmin) / (svg_xmax - svg_xmin)
    rel_y_position = (path_midpoint_y - svg_ymin) / (svg_ymax - svg_ymin)
    return rel_x_position, rel_y_position


def get_relative_pos_to_bounding_box_of_animated_paths(file, animation_id, animated_animation_ids):
    """ Get relative position of a path to the bounding box of all animated paths.

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element.
        animated_animation_ids (list(int)): List of animated element IDs.

    Returns:
        float, float: Relative x- and y-position of path to bounding box of all animated paths.

    """
    path_midpoint_x, path_midpoint_y = get_midpoint_of_path_bbox(file, animation_id)
    xmin, xmax, ymin, ymax = get_bbox_of_multiple_paths(file, animated_animation_ids)
    try:
        rel_x_position = (path_midpoint_x - xmin) / (xmax - xmin)
    except Exception as e1:
        rel_x_position = 0.5
        print(f"{file}, animation_id {animation_id}, animated_animation_ids {animated_animation_ids}: rel_x_position not defined and set to 0.5. {e1}")
    try:
        rel_y_position = (path_midpoint_y - ymin) / (ymax - ymin)
    except Exception as e2:
        rel_y_position = 0.5
        print(f"{file}, animation_id {animation_id}, animated_animation_ids {animated_animation_ids}: rel_y_position not defined and set to 0.5. {e2}")

    return rel_x_position, rel_y_position


def get_relative_path_size(file, animation_id):
    """ Get relative size of a path in an SVG.

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element.

    Returns:
        float, float: Relative width and height of path.

    """
    svg_xmin, svg_xmax, svg_ymin, svg_ymax = get_svg_bbox(file)
    svg_width = float(svg_xmax - svg_xmin)
    svg_height = float(svg_ymax - svg_ymin)

    path_xmin, path_xmax, path_ymin, path_ymax = get_path_bbox(file, animation_id)
    path_width = float(path_xmax - path_xmin)
    path_height = float(path_ymax - path_ymin)

    rel_width = path_width / svg_width
    rel_height = path_height / svg_height

    return rel_width, rel_height


def get_begin_values_by_starting_pos(file, animation_ids, start=1, step=0.5):
    """ Get begin values by sorting from left to right.

    Args:
        file (str): Path of SVG file.
        animation_ids (list(int)): List of element IDs.
        start (float): First begin value.
        step (float): Time between begin values.

    Returns:
        list: Begin values of element IDs.

    """
    starting_point_list = []
    begin_list = []
    begin = start
    for i in range(len(animation_ids)):
        x, _, _, _ = get_path_bbox(file, animation_ids[i])  # get x value of upper left corner
        starting_point_list.append(x)
        begin_list.append(begin)
        begin = begin + step

    animation_id_order = [z for _, z in sorted(zip(starting_point_list, range(len(starting_point_list))))]
    begin_values = [z for _, z in sorted(zip(animation_id_order, begin_list))]

    return begin_values


