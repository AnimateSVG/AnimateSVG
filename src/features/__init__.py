"""
.. automodule:: src.features.create_path_vector
    :members:
.. automodule:: src.features.get_style_attributes
    :members:
.. automodule:: src.features.get_style_attributes_folder
    :members:
.. automodule:: src.features.get_svg_color_tendency
    :members:
.. automodule:: src.features.get_svg_size_pos
    :members:
"""

from .get_style_attributes import get_style_attributes_svg, get_style_attributes_path
from .get_style_attributes_folder import get_style_attributes_folder
from .get_svg_color_tendency import get_svg_color_tendencies
from .get_svg_size_pos import get_svg_size, get_svg_bbox, get_path_bbox, get_midpoint_of_path_bbox, \
    get_bbox_of_multiple_paths, get_relative_path_pos, get_relative_pos_to_bounding_box_of_animated_paths, \
    get_relative_path_size, get_begin_values_by_starting_pos


