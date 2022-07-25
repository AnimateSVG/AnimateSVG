from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from xml.dom import minidom

pd.options.mode.chained_assignment = None  # default='warn'


def get_style_attributes_svg(file):
    """ Get style attributes of an SVG.

    Args:
        file (str): Path of SVG file.

    Returns:
        pd.DataFrame: Dataframe containing the attributes of each path.

    """
    local_styles = get_local_style_attributes(file)
    global_styles = get_global_style_attributes(file)
    global_group_styles = get_global_group_style_attributes(file)
    return combine_style_attributes(local_styles, global_styles, global_group_styles)


def get_style_attributes_path(file, animation_id, attribute):
    """ Get style attributes of a specific path in an SVG.

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element.
        attribute (str): One of the following: fill, stroke, stroke_width, opacity, stroke_opacity.

    Returns:
        str: Attribute of specific path.

    """
    styles = get_style_attributes_svg(file)
    styles_animation_id = styles[styles["animation_id"] == str(animation_id)]
    return styles_animation_id.iloc[0][attribute]


def parse_svg(file):
    """ Parse a SVG file.

    Args:
        file (str): Path of SVG file.

    Returns:
        list, list: List of path objects, list of dictionaries containing the attributes of each path.

    """
    paths, attrs = svg2paths(file)
    return paths, attrs


def get_local_style_attributes(file):
    """ Generate dataframe containing local style attributes of an SVG.

    Args:
        file (str): Path of SVG file.

    Returns:
        pd.DataFrame: Dataframe containing filename, animation_id, class, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    return pd.DataFrame.from_records(_get_local_style_attributes(file))


def _get_local_style_attributes(file):
    try:
        _, attributes = parse_svg(file)
    except:
        print(f'{file}: Attributes not defined.')
    for i, attr in enumerate(attributes):
        animation_id = attr['animation_id']
        class_ = ''
        fill = '#000000'
        stroke = '#000000'
        stroke_width = '0'
        opacity = '1.0'
        stroke_opacity = '1.0'

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

        yield dict(filename=file.split('.svg')[0], animation_id=animation_id, class_=class_, fill=fill, stroke=stroke,
                   stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_style_attributes(file):
    """ Generate dataframe containing global style attributes of an SVG.

    Args:
        file (str): Path of SVG file.

    Returns:
        pd.DataFrame: Dataframe containing filename, class, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    return pd.DataFrame.from_records(_get_global_style_attributes(file))


def _get_global_style_attributes(file):
    doc = minidom.parse(file)
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

            yield dict(filename=file.split('.svg')[0], class_=class_, fill=fill, stroke=stroke,
                       stroke_width=stroke_width, opacity=opacity, stroke_opacity=stroke_opacity)


def get_global_group_style_attributes(file):
    """ Generate dataframe containing global style attributes defined through <g> tags of an SVG.

    Args:
        file (str): Path of SVG file.

    Returns:
        pd.DataFrame: Dataframe containing filename, href, animation_id, fill, stroke, stroke_width, opacity, stroke_opacity.

    """
    df_group_animation_id_matching = pd.DataFrame.from_records(_get_group_animation_id_matching(file))

    df_group_attributes = pd.DataFrame.from_records(_get_global_group_style_attributes(file))
    df_group_attributes.drop_duplicates(inplace=True)
    df_group_attributes.replace("", float("NaN"), inplace=True)
    df_group_attributes.dropna(thresh=3, inplace=True)

    if "href" in df_group_attributes.columns:
        df_group_attributes.dropna(subset=["href"], inplace=True)

    if df_group_attributes.empty:
        return df_group_attributes
    else:
        return df_group_animation_id_matching.merge(df_group_attributes, how='left', on=['filename', 'href'])


def _get_global_group_style_attributes(file):
    doc = minidom.parse(file)
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


def _get_group_animation_id_matching(file):
    doc = minidom.parse(file)
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
            animation_id = defs[i].getElementsByTagName('clipPath')[0].getElementsByTagName('path')[0].getAttribute('animation_id')
            yield dict(filename=file.split('.svg')[0], href=href, animation_id=animation_id)


def combine_style_attributes(df_local, df_global, df_global_groups):
    """ Combine local und global style attributes. Global attributes have priority.

    Args:
        df_local (pd.DataFrame): Dataframe with local style attributes.
        df_global (pd.DataFrame): Dataframe with global style attributes.
        df_global_groups (pd.DataFrame): Dataframe with global style attributes defined through <g> tags.

    Returns:
        pd.DataFrame: Dataframe with all style attributes.

    """
    if df_global.empty and df_global_groups.empty:
        df_local.insert(loc=3, column='href', value="")
        return df_local

    if not df_global.empty:
        df = df_local.merge(df_global, how='left', on=['filename', 'class_'])
        df_styles = df[["filename", "animation_id", "class_"]]
        df_styles["fill"] = _combine_columns(df, "fill")
        df_styles["stroke"] = _combine_columns(df, "stroke")
        df_styles["stroke_width"] = _combine_columns(df, "stroke_width")
        df_styles["opacity"] = _combine_columns(df, "opacity")
        df_styles["stroke_opacity"] = _combine_columns(df, "stroke_opacity")
        df_local = df_styles.copy(deep=True)
    if not df_global_groups.empty:
        df = df_local.merge(df_global_groups, how='left', on=['filename', 'animation_id'])
        df_styles = df[["filename", "animation_id", "class_", "href"]]
        df_styles["href"] = df_styles["href"].fillna('')
        df_styles["fill"] = _combine_columns(df, "fill")
        df_styles["stroke"] = _combine_columns(df, "stroke")
        df_styles["stroke_width"] = _combine_columns(df, "stroke_width")
        df_styles["opacity"] = _combine_columns(df, "opacity")
        df_styles["stroke_opacity"] = _combine_columns(df, "stroke_opacity")

    return df_styles


def _combine_columns(df, col_name):
    col = np.where(~df[f"{col_name}_y"].astype(str).isin(["", "nan"]),
                   df[f"{col_name}_y"], df[f"{col_name}_x"])
    return col


def transform_to_hex(rgb):
    """ Transform RGB to hex.

    Args:
        rgb (str): RGB code.

    Returns:
        str: Hex code.

    """
    if rgb == 'none':
        return '#000000'
    if 'rgb' in rgb:
        rgb = rgb.replace('rgb(', '').replace(')', '')
        if '%' in rgb:
            rgb = rgb.replace('%', '')
            rgb_list = rgb.split(',')
            r_value, g_value, b_value = [int(float(i) / 100 * 255) for i in rgb_list]
        else:
            rgb_list = rgb.split(',')
            r_value, g_value, b_value = [int(float(i)) for i in rgb_list]
        return '#%02x%02x%02x' % (r_value, g_value, b_value)
