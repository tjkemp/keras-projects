Keras Installation
==================

Requirements:
 - 64-bit Python < 3.7

1. To start a totally new Keras project create a virtualenv for it.

.. code-block:: bash

	conda create -n tensorflow python=3.6


Then install tensorflow and Keras and it's dependencies mentioned in requirements.txt.

.. code-block:: bash
	conda install packagename

If you're installing with pip and it can't find tensorflow then install it from 
[https://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow](https://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow).

2. By default Keras uses Theano. To switch to tensorflow, first enter python shell and import Keras to create the configuration file. 

.. code-block:: python

	import keras

Then change the default backend from 'Theano' to 'tensorflow'.

.. code-block:: bash

	nano ~/.keras/keras.json

(On Windows environments the file is C:\Users\UserName\.keras\keras.json.)

Troubleshooting
---------------

If it still doesn’t work make sure you have python development stuff installed:

.. code-block:: bash

	sudo apt-get install python3-dev

If matplotlib says ImportError: "No module named '_tkinter', please install the python3-tk package" then do just that.

.. code-block:: bash

	sudo apt-get install python3-tk

And then install matplotlib with pip.
