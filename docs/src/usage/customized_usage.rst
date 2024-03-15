.. _custom_usage:

Customized Usage Example
==========================

For a basic usage example, see :ref:`basic_usage`.

The environment also provides options for customization.

Custom Map
------------
Work in progress on how to add a custom map.

Random Track Generator
-----------------------

The script `examples/random_trackgen.py` allows to generate random tracks.

To use it, the following dependencies are required:

::

	pip install cv2
	pip install numpy
	pip install shapely
	pip install matplotlib


The script can be run by specifying `seed`, number of maps to generate `n_maps` and the output directory `output_dir`.

For example, to generate 3 random maps and store them in the directory `custom_maps`:

::

	python examples/random_trackgen.py --seed 42 --n-maps 3 --outdir custom_maps


.. image:: ../../../../src/assets/random_trackgen.png
	:width: 800
	:align: center