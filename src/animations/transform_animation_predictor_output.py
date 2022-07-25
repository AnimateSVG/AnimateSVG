from src.features.get_svg_size_pos import get_svg_size, get_midpoint_of_path_bbox
from src.features.get_style_attributes import get_style_attributes_path
from src.features.get_svg_color_tendency import get_svg_color_tendencies


def transform_animation_predictor_output(file, animation_id, output):
    """ Function to translate the numeric model output to animation commands.

    Example: transform_animation_predictor_output("data/svgs/logo_1.svg", 0, [0,0,1,0,0,0,-1,-1,-1,0.42,-1,-1])

    Args:
        file (str): Path of SVG file.
        animation_id (int): ID of element in SVG that gets animated.
        output (list): 12-dimensional list of numeric values of which first 6 determine the animation to be used and
                        the last 6 determine the attribute from. Format: [translate, scale, rotate, skew, fill, opacity, translate_from_1, translate_from_2, scale_from, rotate_from, skew_from_1, skew_from_2].

    Returns:
        dict: Animation statement as dictionary.

    """
    animation = {}
    width, height = get_svg_size(file)
    x_midpoint, y_midpoint = get_midpoint_of_path_bbox(file, animation_id)
    fill_style = get_style_attributes_path(file, animation_id, "fill")
    stroke_style = get_style_attributes_path(file, animation_id, "stroke")
    opacity_style = get_style_attributes_path(file, animation_id, "opacity")
    color_1, color_2 = get_svg_color_tendencies(file)

    if output[0] == 1:
        animation["type"] = "translate"
        x = (output[6] * 2 - 1) * width  # between -width and width
        y = (output[7] * 2 - 1) * height  # between -height and height
        animation["from_"] = f"{str(x)} {str(y)}"
        animation["to"] = "0 0"

    elif output[1] == 1:
        animation["type"] = "scale"
        animation["from_"] = output[8] * 2  # between 0 and 2
        animation["to"] = 1

    elif output[2] == 1:
        animation["type"] = "rotate"
        degree = int(output[9]*720) - 360  # between -360 and 360
        animation["from_"] = f"{str(degree)} {str(x_midpoint)} {str(y_midpoint)}"
        animation["to"] = f"0 {str(x_midpoint)} {str(y_midpoint)}"

    elif output[3] == 1:
        if output[10] > 0.5:
            animation["type"] = "skewX"
            animation["from_"] = (output[11] * 2 - 1) * width/20  # between -width/20 and width/20
        else:
            animation["type"] = "skewY"
            animation["from_"] = (output[11] * 2 - 1) * height/20  # between -height/20 and height/20
        animation["to"] = 0

    elif output[4] == 1:
        animation["type"] = "fill"
        if fill_style == "none" and stroke_style != "none":
            color_hex = stroke_style
        else:
            color_hex = fill_style
        animation["to"] = color_hex

        if color_hex != color_1:
            color_from = color_1
        else:
            color_from = color_2
        animation["from_"] = color_from

    elif output[5] == 1:
        animation["type"] = "opacity"
        animation["from_"] = 0
        animation["to"] = opacity_style

    animation["dur"] = 4
    animation["begin"] = 1
    animation["fill"] = "freeze"

    return animation
