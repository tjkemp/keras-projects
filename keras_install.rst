Keras Installation
=================

To start a Keras project create a virtualenv for it and install keras.

.. code-block:: bash

	mkproject keras-test
	pip install keras

Curiously by default it uses tensorflow but installs Theano. To fix it, try to import keras once and then change the default engine with

.. code-block:: bash

	nano ~/.keras.keras.json

To do this enter python shell and try to import something from Keras. You’ll get “ImportError: No module named 'tensorflow'”

.. code-block:: python

	from keras.datasets import mnist

Edit the configuration file and replace “tensorflow” with “theano”.

.. code-block:: bash

	nano ~/.keras/keras.json

It still doesn’t work unless you got all the python development stuff installed.

.. code-block:: bash

	sudo apt-get install python3-dev

To visualize the stuff we need matplotlib but installing it doesn’t work out of the box either. It will raise an ImportError: “No
module named '_tkinter', please install the python3-tk package”.

To fix it, type:

.. code-block:: bash

	sudo apt-get install python3-tk

And then install matplotlib.

.. code-block:: bash

	pip install matplotlib
