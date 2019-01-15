import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
import common

# define parameters
batch_size = 64
num_epochs = 60
validation_split = 0.16666
num_classes = 10

def main():

    # load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # calculate sizes
    samples_train_num = x_train.shape[0]
    samples_validation_num = round(samples_train_num * validation_split)
    samples_test_num = x_test.shape[0]

    # flatten data
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # rescale data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert targets to binary classes
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print("{} samples in training set (of which {} are used for validation).".format(
        samples_train_num, samples_validation_num))
    print("{} samples in test test.".format(
        samples_test_num))

    # 1) define the model
    model = Sequential()
    model.add(Dense(28, activation='relu', input_shape=(784,)))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='SGD', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()

    # 2) train the model
    history = model.fit(x_train, y_train, 
        validation_split=validation_split, 
        epochs=num_epochs, 
        batch_size=batch_size)

    # 3) show results
    score = model.evaluate(x_test, y_test, verbose=0)    
    print('Final score:', score[0])
    print('Final accuracy:', score[1])

    common.plot_history(history)

if __name__ == "__main__":
    main()
