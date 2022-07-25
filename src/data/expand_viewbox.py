from xml.dom import minidom
from pathlib import Path
from svgpathtools import svg2paths
import os


def expand_viewbox_in_folder(old_folder, percent=50, new_folder="data/expanded_svgs"):
    """ Expand the viewboxes of all SVGs in a given folder.

    Args:
        old_folder (str): Path of folder containing all SVG files.
        percent (int): Percentage in %: How much do we want to expand the viewbox? Default is 50%.
        new_folder (str): Path of folder containing the expanded SVGs.

    """
    for file in os.listdir(old_folder):
        if file.endswith(".svg"):
            expand_viewbox(old_folder + "/" + file, percent, new_folder)


def expand_viewbox(svg_file, percent=50, new_folder="data/expanded_svgs"):
    """ Expand the viewbox of a given SVG.

    Args:
        svg_file (svg): Path of SVG file.
        percent (int): Percentage in %: How much do we want to expand the viewbox? Default is 50%.
        new_folder (str): Path of folder containing the expanded SVGs.

    """
    Path(new_folder).mkdir(parents=True, exist_ok=True)
    pathelements = svg_file.split('/')
    filename = pathelements[len(pathelements) - 1].replace('.svg', '')

    doc = minidom.parse(svg_file)
    x, y = '', ''
    # get width and height of logo
    try:
        width = doc.getElementsByTagName('svg')[0].getAttribute('width')
        height = doc.getElementsByTagName('svg')[0].getAttribute('height')
        if not width[-1].isdigit():
            width = width.replace('px', '').replace('pt', '')
        if not height[-1].isdigit():
            height = height.replace('px', '').replace('pt', '')
        x = float(width)
        y = float(height)
        check = True
    except:
        check = False
    if not check:
        try:
            # get bounding box of svg
            xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
            paths, attributes = svg2paths(svg_file)
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
                x = xmax_svg - xmin_svg
                y = ymax_svg - ymin_svg
        except:
            print('Error: ' + filename)
            return
    # Check if viewBox exists
    if doc.getElementsByTagName('svg')[0].getAttribute('viewBox') == '':
        v1, v2, v3, v4 = 0, 0, 0, 0
        # Calculate new viewBox values
        x_new = x * (100 + percent) / 100
        y_new = y * (100 + percent) / 100
    else:
        v1 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[0].replace('px', '').replace('pt', '').replace(',', ''))
        v2 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[1].replace('px', '').replace('pt', '').replace(',', ''))
        v3 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[2].replace('px', '').replace('pt', '').replace(',', ''))
        v4 = float(doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[3].replace('px', '').replace('pt', '').replace(',', ''))
        x = v3
        y = v4
        # Calculate new viewBox values
        x_new = x * percent / 100
        y_new = y * percent / 100
    x_translate = - x * percent / 200
    y_translate = - y * percent / 200
    coordinates = str(v1 + x_translate) + ' ' + str(v2 + y_translate) + ' ' + str(v3 + x_new) + ' ' + str(v4 + y_new)
    doc.getElementsByTagName('svg')[0].setAttribute('viewBox', coordinates)
    # write to svg
    textfile = open(new_folder + '/' + filename + '.svg', 'wb')
    textfile.write(doc.toprettyxml(encoding="iso-8859-1"))  # needed to handle "Umlaute"
    textfile.close()
    doc.unlink()
