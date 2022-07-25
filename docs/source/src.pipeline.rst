Logo Pipeline
=============

    You can create animations of an SVG logo by using the following lines of code:

        >>> from src.pipeline import Logo
        >>> logo = Logo(data_dir='path/to/my/svgs/logo.svg')
        >>> logo.animate()

    Two animations are saved to the directory path/to/my/svgs, one built with a genetic algorithm (GA) and one
    built with an entmoot optimizer. If you only want to use one of the two, you can specify that as follows:

        >>> logo.animate(model='ga')
        >>> logo.animate(model='opt')

    Printing logo information such as total number of elements, size and bounding box works as follows:

        >>> logo.print_logo_information()


.. autoclass:: src.pipeline.Logo
    :members:

