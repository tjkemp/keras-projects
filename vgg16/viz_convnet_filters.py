from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

model = VGG16(weights='imagenet', include_top=False)
model.summary()

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150, iterations=20):
    """ Creates a visual pattern that each filter is maximally responsive to.

    This is done by applying gradient descent to the value of the input
    image of a convnet so as to maximize the response of a specific filter.
    """
    input_img_data = np.random.random((1, 128, 128, 3)) * 20 + 128.

    # creates a loss function that maximises the activation of the filter
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt to the loss
    grads = K.gradients(loss, model.input)[0]
    # a trick to help gradient descent to go smoothly is to normalize the tensor by its L2
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # creates a function which returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # compute value of the loss tensor and the gradient tensor given an input image
    # iterate takes a numpy tensor (as list of tensors size 1) and returns a list of two
    # numpy tensors: the loss value and the gradient value

    # loss maximization via sgd
    step = 1.
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern('block3_conv1', 1, iterations=20))
plt.show()
