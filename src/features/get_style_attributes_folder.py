from src.features.get_style_attributes import combine_style_attributes, transform_to_hex, parse_svg
import pandas as pd
from xml.dom import minidom
import os

pd.options.mode.chained_assignment = None  # default='warn'


def get_style_attributes_folder(folder):
    """ Get style attributes of all SVGs in a given folder.

    Args:
        folder (str): Path of folder containing all SVGs.

    Returns:
        pd.DataFrame: Dataframe containing the attributes of each path of all SVGs.

    """
    local_styles = get_local_style_attributes(folder)
    global_styles = get_global_style_attributes(folder)
    global_group_styles = get_global_group_style_attributes(folder)
    return combine_style_attributes(local_styles, global_styles, global_group_styles)


def get_local_style_attributes(folder):
    """ Generate dataframe containing local style attributes of all SVGs in a given folder.

    Args:
        folder (str): Path of folder containing all SVGs.

    Returns:
        pd.DataFrame: Dataframe containing filename, animation_id, class, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    return pd.DataFrame.from_records(_get_local_style_attributes(folder))


def _get_local_style_attributes(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            try:
                _, attributes = parse_svg(folder + '/' + file)
            except:
                print(f'{file}: Attributes not defined.')
            for i, attr in enumerate(attributes):
                animation_id = attr['animation_id']
                fill = '#000000'
                stroke = '#000000'
                stroke_width = '0'
                opacity = '1.0'
                stroke_opacity = '1.0'
                class_ = ''
                if 'style' in attr:
                    a = attr['style']
                    if a.find('fill') != -1:
                        fill = a.split('fill:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke') != -1:
                        stroke = a.split('stroke:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke-width') != -1:
                        stroke_width = a.split('stroke-width:', 1)[-1].split(';', 1)[0]
                    if a.find('opacity') != -1:
                        opacity = a.split('opacity:', 1)[-1].split(';', 1)[0]
                    if a.find('stroke-opacity') != -1:
                        stroke_opacity = a.split('stroke-opacity:', 1)[-1].split(';', 1)[0]
                else:
                    if 'fill' in attr:
                        fill = attr['fill']
                    if 'stroke' in attr:
                        stroke = attr['stroke']
                    if 'stroke-width' in attr:
                        stroke_width = attr['stroke-width']
                    if 'opacity' in attr:
                        opacity = attr['opacity']
                    if 'stroke-opacity' in attr:
                        stroke_opacity = attr['stroke-opacity']

                if 'class' in attr:
                    class_ = attr['class']

                # transform None and RGB to hex
                if '#' not in fill and fill != '':
                    fill = transform_to_hex(fill)
                if '#' not in stroke and stroke != '':
                    stroke = transform_to_hex(stroke)

                # Cannot handle colors defined with linearGradient
                if 'url' in fill:
                    fill = '#000000'

                yield dict(filename=file.split('.svg')[0], animation_id=animation_id, class_=class_, fill=fill,
                           stroke=stroke, stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_style_attributes(folder):
    """ Generate dataframe containing global style attributes of all SVGs in a given folder.

    Args:
        folder (str): Path of folder containing all SVGs.

    Returns:
        pd.DataFrame: Dataframe containing filename, class, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    return pd.DataFrame.from_records(_get_global_style_attributes(folder))


def _get_global_style_attributes(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            doc = minidom.parse(folder + '/' + file)
            style = doc.getElementsByTagName('style')
            for i, attr in enumerate(style):
                a = attr.toxml()
                for j in range(0, len(a.split(';}')) - 1):
                    fill = ''
                    stroke = ''
                    stroke_width = ''
                    opacity = ''
                    stroke_opacity = ''
                    attr = a.split(';}')[j]
                    class_ = attr.split('.', 1)[-1].split('{', 1)[0]
                    if attr.find('fill:') != -1:
                        fill = attr.split('fill:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke:') != -1:
                        stroke = attr.split('stroke:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke-width:') != -1:
                        stroke_width = attr.split('stroke-width:', 1)[-1].split(';', 1)[0]
                    if attr.find('opacity:') != -1:
                        opacity = attr.split('opacity:', 1)[-1].split(';', 1)[0]
                    if attr.find('stroke-opacity:') != -1:
                        stroke_opacity = attr.split('stroke-opacity:', 1)[-1].split(';', 1)[0]

                    # transform None and RGB to hex
                    if '#' not in fill and fill != '':
                        fill = transform_to_hex(fill)
                    if '#' not in stroke and stroke != '':
                        stroke = transform_to_hex(stroke)

                    # Cannot handle colors defined with linearGradient
                    if 'url' in fill:
                        fill = ''

                    yield dict(filename=file.split('.svg')[0], class_=class_, fill=fill, stroke=stroke,
                               stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_group_style_attributes(folder):
    """ Generate dataframe containing global style attributes defined through <g> tags of all SVGs in a given folder.

    Args:
        folder (str): Path of folder containing all SVG file.

    Returns:
        pd.DataFrame: Dataframe containing filename, href, animation_id, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    df_group_animation_id_matching = pd.DataFrame.from_records(_get_group_animation_id_matching(folder))

    df_group_attributes = pd.DataFrame.from_records(_get_global_group_style_attributes(folder))
    df_group_attributes.drop_duplicates(inplace=True)

    df_group_attributes.replace("", float("NaN"), inplace=True)
    df_group_attributes.dropna(subset=["href"], inplace=True)

    if df_group_attributes.empty:
        return df_group_attributes
    else:
        return df_group_animation_id_matching.merge(df_group_attributes, how='left', on=['filename', 'href'])


def _get_global_group_style_attributes(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            doc = minidom.parse(folder + '/' + file)
            groups = doc.getElementsByTagName('g')
            for i, _ in enumerate(groups):
                style = groups[i].getAttribute('style')
                href = ''
                fill = ''
                stroke = ''
                stroke_width = ''
                opacity = ''
                stroke_opacity = ''
                if len(groups[i].getElementsByTagName('use')) != 0:
                    href = groups[i].getElementsByTagName('use')[0].getAttribute('xlink:href')
                    if style != '':
                        attributes = style.split(';')
                        for j, _ in enumerate(attributes):
                            attr = attributes[j]
                            if attr.find('fill:') != -1:
                                fill = attr.split('fill:', 1)[-1].split(';', 1)[0]
                            if attr.find('stroke:') != -1:
                                stroke = attr.split('stroke:', 1)[-1].split(';', 1)[0]
                            if attr.find('stroke-width:') != -1:
                                stroke_width = attr.split('stroke-width:', 1)[-1].split(';', 1)[0]
                            if attr.find('opacity:') != -1:
                                opacity = attr.split('opacity:', 1)[-1].split(';', 1)[0]
                            if attr.find('stroke-opacity:') != -1:
                                stroke_opacity = attr.split('stroke-opacity:', 1)[-1].split(';', 1)[0]
                    else:
                        fill = groups[i].getAttribute('fill')
                        stroke = groups[i].getAttribute('stroke')
                        stroke_width = groups[i].getAttribute('stroke-width')
                        opacity = groups[i].getAttribute('opacity')
                        stroke_opacity = groups[i].getAttribute('stroke-opacity')

                # transform None and RGB to hex
                if '#' not in fill and fill != '':
                    fill = transform_to_hex(fill)
                if '#' not in stroke and stroke != '':
                    stroke = transform_to_hex(stroke)

                yield dict(filename=file.split('.svg')[0], href=href.replace('#', ''), fill=fill, stroke=stroke,
                           stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def _get_group_animation_id_matching(folder):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            doc = minidom.parse(folder + '/' + file)
            try:
                symbol = doc.getElementsByTagName('symbol')
                for i, _ in enumerate(symbol):
                    href = symbol[i].getAttribute('id')
                    animation_id = symbol[i].getElementsByTagName('path')[0].getAttribute('animation_id')
                    yield dict(filename=file.split('.svg')[0], href=href, animation_id=animation_id)
            except:
                defs = doc.getElementsByTagName('defs')
                for i, _ in enumerate(defs):
                    href = defs[i].getElementsByTagName('symbol')[0].getAttribute('id')
                    animation_id = defs[i].getElementsByTagName('clipPath')[0].getElementsByTagName('path')[
                        0].getAttribute('animation_id')
                    yield dict(filename=file.split('.svg')[0], href=href, animation_id=animation_id)



