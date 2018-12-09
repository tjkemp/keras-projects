import numpy as np
import matplotlib.pyplot as plt

def make_prediction(X_test, y_test, example_id):
    """ Makes a prediction on a single item and prints the result. """

    example = np.array(X_test[example_id,:,:]).reshape((1, 28, 28))
    Yhat = model.predict(example)
    
    print("Testing a test set element id {}: it is {} and the model predicted it to be {}.".format(
        example_id, np.argmax(y_test[example_id]), np.argmax(Yhat)))

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
