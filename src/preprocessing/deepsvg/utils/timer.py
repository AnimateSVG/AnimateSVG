"""This code is taken from <https://github.com/alexandre01/deepsvg>
by Alexandre Carlier, Martin Danelljan, Alexandre Alahi and Radu Timofte
from the paper >https://arxiv.org/pdf/2007.11301.pdf>
"""

import time


class Timer:
    def __init__(self):
        self.time = time.time()

    def reset(self):
        self.time = time.time()

    def get_elapsed_time(self):
        elapsed_time = time.time() - self.time
        self.reset()
        return elapsed_time
