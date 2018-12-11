# Keras for MNIST data set

The task is to rain a model to recognize the MNIST data set and plot accuracy.

## Usage

The simple model (28 neuros) can be trained with:

```sh

python train_small.py

```

and the bigger (512 neurons per layer and using dropout) with:

```sh

python train_wdropout.py

```


## An example run

```sh

	(keras-projects) /Projects/keras-projects/mnist$ python train_simple.py
	
	60000 samples in training set (of which 10000 are used for validation).
	10000 samples in test test.
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	dense_1 (Dense)              (None, 28)                21980
	_________________________________________________________________
	dense_2 (Dense)              (None, 28)                812
	_________________________________________________________________
	dense_3 (Dense)              (None, 10)                290
	=================================================================
	Total params: 23,082
	Trainable params: 23,082
	Non-trainable params: 0
	_________________________________________________________________
	Train on 50000 samples, validate on 10000 samples
	Epoch 1/60
	2018-12-09 15:38:09.720582: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
	50000/50000 [==============================] - 2s 39us/step - loss: 1.3469 - acc: 0.6095 - val_loss: 0.6148 - val_acc: 0.8446
	Epoch 2/60
	50000/50000 [==============================] - 2s 35us/step - loss: 0.5004 - acc: 0.8648 - val_loss: 0.3805 - val_acc: 0.8943
	Epoch 3/60
	.
	.
	50000/50000 [==============================] - 2s 34us/step - loss: 0.0831 - acc: 0.9766 - val_loss: 0.1247 - val_acc: 0.9642
	Final score: 0.12460193493627011
	Final accuracy: 0.9622
```

## Results

The models are 2 hidden layers deep. The example shows the details of the model.

50000 images are used for training, 10000 for development (validation) and 10000 for testing for final accuracy.

### 2 hidden layer, 28 neuron per layer model

Right of the bat with a small non-optimized model we get an accuracy of about 96,5% which is awesome. No sweating
with feature engineering either.

![28 neuron model](images/model_28n.png)

### 2 hidden layer, 512 neuron per layer model (with dropout)

We get final accuracy of 98.04% after 20 epochs.

The bigger model takes considerably more time to train without a GPU. 

![512 neuron model](images/model_512n.png)

### 2 hidden layer, 512 neuron per layer model (without dropout)

Final accuracy of 98.36% after 20 epochs.

![512 neuron model with dropout](images/model_512n_wdropout.png)
