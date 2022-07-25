"""This code is taken from <https://github.com/alexandre01/deepsvg>
by Alexandre Carlier, Martin Danelljan, Alexandre Alahi and Radu Timofte
from the paper >https://arxiv.org/pdf/2007.11301.pdf>
"""

from .default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 20 #60 * num_gpus

        self.val_every = 10 #2000
