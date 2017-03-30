*********************************************************************
Keras for MNIST data set, accuracy plot and model evaluation
*********************************************************************

The task
===========

Train a model in Keras to recognize the MNIST data set, plot the accuracy and evaluate the model.
The model is 3 hidden layers deep. The run example shows the details of the model.

With this non-optimized model we get an accuracy of about 96,5%.
The model started to overfit after 50 epochs.


An example run
==================

.. code-block:: bash

	(keras-projects) tero@Ubik:~/Projects/keras-projects/mnist$ python train.py
	Using Theano backend.
	Input: 60000 samples in training set (of which 9960 are used for validation) and 10000 samples in test test.
	____________________________________________________________________________________________________
	Layer (type) Output Shape Param # Connected to
	====================================================================================================
	dense_1 (Dense) (64, 28, 28) 812 dense_input_1[0][0]
	____________________________________________________________________________________________________
	dense_2 (Dense) (64, 28, 28) 812 dense_1[0][0]
	____________________________________________________________________________________________________
	flatten_1 (Flatten) (64, 784) 0 dense_2[0][0]
	____________________________________________________________________________________________________
	dense_3 (Dense) (64, 10) 7850 flatten_1[0][0]
	====================================================================================================
	Total params: 9,474
	Trainable params: 9,474
	Non-trainable params: 0
	____________________________________________________________________________________________________
	Train on 50040 samples, validate on 9960 samples
	Epoch 1/120
	50040/50040 [==============================] - 2s - loss: 1.3150 - acc: 0.6287 - val_loss: 0.5515 - val_acc: 0.8499
	.
	.
	.
	Epoch 120/120
	50040/50040 [==============================] - 4s - loss: 0.0853 - acc: 0.9743 - val_loss: 0.1158 - val_acc: 0.9689
	Final score: 0.116527532629
	Final accuracy: 0.965

