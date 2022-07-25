"""
.. automodule:: src.animations.create_random_animations
    :members:
.. automodule:: src.animations.get_path_probabilities
    :members:
.. automodule:: src.animations.insert_animation
    :members:
.. automodule:: src.animations.transform_animation_predictor_output
    :members:
"""

from .create_random_animations import create_random_animations
from .get_path_probabilities import get_path_probabilities
from .insert_animation import create_animated_svg, insert_translate_statement, insert_scale_statement, \
    insert_rotate_statement, insert_skew_statement, insert_fill_statement, insert_opacity_statement
from .transform_animation_predictor_output import transform_animation_predictor_output
