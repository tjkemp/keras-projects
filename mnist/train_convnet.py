import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import common

# define parameters
batch_size = 64
num_epochs = 5
validation_split = 0.16666
num_classes = 10

def main():

    # load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # limit input size
    #x_train = x_train[:200]
    #y_train = y_train[:200]
    #x_test = x_test[:50]
    #y_test = y_test[:50]

    # reshape and rescale
    print(x_train.shape)
    train_size, _, _ = x_train.shape
    train_images = x_train.reshape((train_size, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_size, _, _ = x_test.shape
    test_images = x_test.reshape((test_size, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    # convert targets to binary classes
    train_labels = np_utils.to_categorical(y_train, num_classes)
    test_labels = np_utils.to_categorical(y_test, num_classes)

    # 1) define the model
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()

    # 2) train the model
    history = model.fit(train_images, train_labels, 
        validation_split=validation_split, 
        epochs=num_epochs, 
        batch_size=batch_size)

    # 3) show results
    score = model.evaluate(test_images, test_labels)    
    print('Accuracy after last epoch:', score[1])

    common.plot_history(history)

if __name__ == "__main__":
    main()
