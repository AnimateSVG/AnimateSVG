from xml.dom import minidom
from pathlib import Path
import os


""" 
In order for the function to work properly, an animation_id first has to be inserted into each element of the SVG.
Use for this the function insert_ids.py prior to this function.
"""


def decompose_logos_in_folder(folder):
    """ Decompose all SVGs in a folder.

    Args:
        folder (str): Path of folder containing all SVGs that need to be decomposed.

    """
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            decompose_logo(folder + "/" + file)


def decompose_logo(file):
    """ Decompose one SVG.

    Args:
        file (str): Path of SVG that needs to be decomposed.

    """
    doc = minidom.parse(file)
    # store all elements in list
    elements = doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
        'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName('polygon') + doc.getElementsByTagName(
        'polyline') + doc.getElementsByTagName('rect') + doc.getElementsByTagName('text')
    num_elements = len(elements)

    # Change name and path for writing element svgs
    filename = file.split('/')[-1].replace('.svg', '')
    Path("data/decomposed_svgs").mkdir(parents=True, exist_ok=True)

    # Write each element to a svg file
    for i in range(num_elements):
        # load svg again: necessary because we delete elements in each loop
        doc_temp = minidom.parse(file)
        elements_temp = doc_temp.getElementsByTagName('path') + doc_temp.getElementsByTagName(
            'circle') + doc_temp.getElementsByTagName('ellipse') + doc_temp.getElementsByTagName(
            'line') + doc_temp.getElementsByTagName('polygon') + doc_temp.getElementsByTagName(
            'polyline') + doc_temp.getElementsByTagName('rect') + doc_temp.getElementsByTagName('text')
        # select all elements besides one
        elements_temp_remove = elements_temp[:i] + elements_temp[i + 1:]
        for element in elements_temp_remove:
            # Check if current element is referenced clip path
            if not element.parentNode.nodeName == "clipPath":
                parent = element.parentNode
                parent.removeChild(element)
        # Add outline to element (to handle white elements on white background)
        elements_temp[i].setAttribute('stroke', 'black')
        elements_temp[i].setAttribute('stroke-width', '2')
        # If there is a style attribute, remove stroke:none
        if len(elements_temp[i].getAttribute('style')) > 0:
            elements_temp[i].attributes['style'].value = elements_temp[i].attributes['style'].value.replace('stroke:none', '')
        # save element svgs
        animation_id = elements_temp[i].getAttribute('animation_id')
        textfile = open('data/decomposed_svgs/' + filename + '_' + animation_id + '.svg', 'wb')
        textfile.write(doc_temp.toxml(encoding="iso-8859-1")) # needed to handle "Umlaute"
        textfile.close()
        doc_temp.unlink()

    doc.unlink()
