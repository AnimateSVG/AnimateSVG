SVG Logo Animation using Machine Learning
-----------------------------------------

This project allows to automatically animate logos in SVG format using machine learning.

Its functionality includes extracting SVG information (e.g., size, position, color), get SVG embeddings of logos by using  `DeepSVG <https://github.com/alexandre01/deepsvg/>`__'s hierarchical generative network, and an entire pipeline that takes an unprocessed logo as input and outputs an animated logo created with two different machine learning models.


Table of Contents
#################

.. contents::

Description
#################

In the light of the constant battle for attention on digital media, animating digital content plays an increasing role in modern graphic design. In this study, we use artificial intelligence methods to create aesthetic animations along the case of brand logos. With scalable vector graphics as the standard format in modern graphic design, we develop an autonomous end-to-end method using complex machine learning techniques to create brand logo animations as scalable vector graphics from scratch. We acquire data and setup a comprehensive animation space to create novel animations and evaluate them based on their aesthetics. We propose and compare two alternative computational models for automated logo animation and carefully weigh up their idiosyncrasies: on the one hand, we set up an aesthetics evaluation model to train an animation generator and, on the other hand, we combine tree ensembles with global optimization. Indeed, our proposed methods are capable of creating aesthetic logo animations, receiving an average rating of ‘good’ from observers.

At the bottom of this documentation you can find some animations that have been generated using our models.


How to Install
##############

To use this code you have to follow these steps:

1. Start by cloning this Git repository:

.. code-block::

    $  git clone https://github.com/AnimateSVG/AnimateSVG.git
    $  cd animate_logos

2. Continue by creating a new conda environment (Python 3.7):

.. code-block::

    $  conda create -n animate_logos python=3.7
    $  conda activate animate_logos

3. Install the dependencies:

.. code-block::

    $ pip install -r requirements.txt
    $ conda install -c conda-forge cairosvg
    $ conda install -c conda-forge lightgbm
    
4. Set the conda environment on your jupyter notebook:

.. code-block::

    $ python -m ipykernel install --user --name=animate_logos 

If there are problems with cairosvg please refer to `this guide <https://cairosvg.org/documentation/#installation/>`__. In case you encounter problems running notebooks 5a and 5b (e.g., the kernel dies), try to install PyTorch Nightly, which may fix the bug:

.. code-block::

    $ conda install pytorch torchvision torchaudio -c pytorch-nightly 

For training our optimization model for the generation of logo animations, we use the commercial `Gurobi Optimizer <https://www.gurobi.com/>`__
with a `free academic licence <https://www.gurobi.com/academia/academic-program-and-licenses/>`__. You can find a detailed
guide to install Gurobi on your computer `here <https://www.gurobi.com/documentation/9.1/quickstart_mac/software_installation_guid.html#section:Installation/>`__.

For completeness, the `labeling website <https://animate-logos.web.app/>`__ where users can rate the quality of animations, is needed.


How to Use
##########

This repository is a documentation of our logo animation system and can be used to track our results and serves as a basis for further research. 

Since we are not allowed to publish our underlying raw logo data due to copyrights, not all notebooks can be rerun (notebooks 2 - 5b under *./notebooks*). However, they make our preprocessing and training steps transparent. 

To test our trained models, you can upload your own logo to *data/uploads* and use notebook *notebooks/6_logo_animation_pipeline.ipynb* or the following code to create your own animations:

.. code:: python

    from src.pipeline import Logo
    logo = Logo(data_dir='path/to/my/svgs/logo.svg')
    logo.animate()


Reference
#########

To get an embedding of SVG logos, we used an approach described by Alexandre Carlier, Martin Danelljan, Alexandre Alahi and Radu Timofte in their paper `DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation <https://arxiv.org/abs/2007.11301>`__ by using the code from this `repository <https://github.com/alexandre01/deepsvg/>`__. You can find the code in the directories src.preprocessing.configs and src.preprocessing.deepsvg.


Examples
#################

In the following you can see some examples of animations that have been generated using our models.

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/23_logo.gif
     - .. image:: data/examples/gif/12_logo.gif
     - .. image:: data/examples/gif/13_logo.gif

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/14_logo.gif
     - .. image:: data/examples/gif/15_logo.gif
     - .. image:: data/examples/gif/16_logo.gif

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/17_logo.gif
     - .. image:: data/examples/gif/30_logo.gif
     - .. image:: data/examples/gif/19_logo.gif

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/20_logo.gif
     - .. image:: data/examples/gif/0_logo.gif
     - .. image:: data/examples/gif/22_logo.gif

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/21_logo.gif
     - .. image:: data/examples/gif/24_logo.gif
     - .. image:: data/examples/gif/25_logo.gif

.. list-table::
   :class: borderless

   * - .. image:: data/examples/gif/29_logo.gif
     - .. image:: data/examples/gif/27_logo.gif
     - .. image:: data/examples/gif/28_logo.gif

.. image:: data/examples/gif/1_logo.gif
.. image:: data/examples/gif/2_logo.gif
.. image:: data/examples/gif/3_logo.gif
.. image:: data/examples/gif/4_logo.gif
.. image:: data/examples/gif/5_logo.gif
.. image:: data/examples/gif/6_logo.gif
.. image:: data/examples/gif/7_logo.gif
.. image:: data/examples/gif/8_logo.gif
.. image:: data/examples/gif/9_logo.gif
.. image:: data/examples/gif/11_logo.gif
