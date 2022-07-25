"""
.. automodule:: src.data.expand_viewbox
    :members:
.. automodule:: src.data.get_svg_meta_data
    :members:
.. automodule:: src.data.insert_ids
    :members:
.. automodule:: src.data.rename_logos
    :members:
.. automodule:: src.data.sm_dataloader
    :members:
.. automodule:: src.data.svg_scraper
    :members:
.. automodule:: src.data.svg_to_png
    :members:
"""
from .expand_viewbox import expand_viewbox_in_folder, expand_viewbox
from .get_svg_meta_data import get_svg_meta_data
from .insert_ids import insert_ids_in_folder
from .rename_logos import rename_logos
from .sm_dataloader import DatasetSM
from .svg_to_png import convert_svgs_in_folder
