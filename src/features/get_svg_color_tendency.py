from src.features.get_style_attributes import get_style_attributes_svg


def get_svg_color_tendencies(file):
    """ Get two most frequent colors in SVG. Black and white are excluded.

    Args:
        file (str): Path of SVG file.

    Returns:
        list: List of two most frequent colors in SVG.

    """
    df = get_style_attributes_svg(file)
    df = df[~df['fill'].isin(['#FFFFFF', '#ffffff'])]
    colour_tendencies_list = df["fill"].value_counts()[:2].index.tolist()
    colour_tendencies_list.append("#000000")
    return colour_tendencies_list[:2]

