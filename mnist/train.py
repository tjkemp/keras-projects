import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import Callback
import matplotlib.pyplot as plt

# define parameters
batch_size = 64
num_epochs = 120
validation_split = 0.166

# load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# calculate sizes
nTrainSamples, xsize, ysize = X_train.shape
nValidationSamples = round(nTrainSamples*validation_split)
nTestSamples = X_test.shape[0]
input_shape = (xsize, ysize)

# rescale data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert targets to binary class matrices
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print("Input: {} samples in training set (of which {} are used for validation) and {} samples in test test.".format(
nTrainSamples, nValidationSamples, nTestSamples))

# 1) define model
# create a sequence and add layers to it in order to perform computation
model = Sequential()
model.add(Dense(28, activation='relu', batch_input_shape=(batch_size, 28, 28)))
model.add(Dense(28, activation='relu', input_dim=(28,28)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# 2) compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# 3) train the model
history = model.fit(X_train, y_train, validation_split=validation_split, nb_epoch=num_epochs, batch_size=batch_size)

# UNCOMMENT the following lines to make a prediction on a single example
#exampleId = 0
#example = np.array(X_test[exampleId,:,:]).reshape((1, 28,28))
#Yhat = model.predict(example)
#print("Testing a test set element id {}: it is {} and the model predicted it to be {}.".format(exampleId,
np.argmax(y_test[exampleId]), np.argmax(Yhat)))

# show results
score = model.evaluate(X_test, y_test, verbose=0)
print('Final score:', score[0])
print('Final accuracy:', score[1])
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Accuracy")
plt.plot(history.history['acc'], label="Training data")
plt.plot(history.history['val_acc'], label="Validation data")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.title("Losses")
plt.plot(history.history['loss'], label="Training data")
plt.plot(history.history['val_loss'], label="Validation data")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
