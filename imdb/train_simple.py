import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
import matplotlib.pyplot as plt


max_features = 5000
batch_size = 512
epochs = 10

def plot_history(history, filename=None, show=True):
    """ Plots accuracy and loss. """

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

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

def word_decoder(train_data):
    """ Returns a function that can return decoded imdb review. """
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    def reverse(idx):
        return " ".join([reverse_word_index.get(i - 3, '?') for i in train_data[idx]])
    return reverse

def max_value(np_array):
    return max([max(item) for item in np_array])

def vectorize_sequence(sequences, dimension):
    """ Turns sequences into vectors of 0s and 1s. """
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

def main():

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    dims = max_value(x_train)+1
    print('with dimension', dims)
    print()
    print('One-hot encode your lists to turn them into 0s and 1s...')
    x_train = vectorize_sequence(x_train, max_features)
    x_test = vectorize_sequence(x_test, max_features)
    print('x_train:', x_train.shape)
    print('x_test:', x_test.shape)

    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    # separate training data into training and validation sets
    x_val = x_train[:10000]
    x_partial_train = x_train[10000:]
    y_val = y_train[:10000]
    y_partial_train = y_train[10000:]

    print('Building model...')
    model = Sequential()

    model.add(Dense(16, activation='relu', input_shape=(max_features,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    
    history = model.fit(x_partial_train, y_partial_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val))

    plot_history(history)

if __name__ == "__main__":
    main()
