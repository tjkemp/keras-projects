from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
import matplotlib.pyplot as plt

max_features = 500
maxlen = 400

embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250

batch_size = 32
epochs = 2

def plot_history(history, filename=None, show=True):
    """ Plots accuracy and loss. """

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Accuracy")
    plt.plot(history.history['acc'], label="Training data")
    plt.plot(history.history['val_acc'], label="Validation data")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    #plt.legend()

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

def main():

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences...')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train:', x_train.shape)
    print('x_test:', x_test.shape)

    print('Building model...')
    model = Sequential()

    # start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # learn word group filters of size filter_length:
    model.add(Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test))

    plot_history(history, filename="images/imdb.png")

if __name__ == "__main__":
    main()
