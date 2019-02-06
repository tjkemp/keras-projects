from keras.applications import VGG16
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import math

model = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150, 150, 3))

# get an image and feed it through the network
imgfile = 'cat.13.jpg'

img = image.load_img(imgfile, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

layer_outputs = [layer.output for layer in model.layers[1:9]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activation_model.summary()

activations = activation_model.predict(img_tensor)

def show_filter(activations, num_layer, num_filter):
    print(activations[num_layer].shape)
    plt.matshow(activations[num_layer][0,:,:,num_filter], cmap='viridis')
    plt.show()

# to see a single activation:
#show_filter(activations, 7, 13)

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    num_features = layer_activation.shape[-1]
    size_pixels = layer_activation.shape[1]

    size_images = round(math.sqrt(num_features + 1))
    #images_per_col = num_features // images_per_row + 1
    display_grid = np.zeros((size_pixels * size_images, size_pixels * size_images))

    for col in range(size_images):
        for row in range(size_images):
            feature = col * size_images + row
            #print(feature)
            try:
                channel_image = layer_activation[0, :, :, feature]
            except IndexError:
                print("IndexError")
                continue
            channel_image -= channel_image.mean()
            std = channel_image.std()
            if std > 0:
                channel_image /= channel_image.std()
                channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            #print([col*size, (col+1)*size, row * size, (row + 1) * size])
            #print(display_grid[col * size : (col + 1) * size,
            #            row * size : (row + 1) * size].shape)
            try:
                display_grid[col * size_pixels : (col + 1) * size_pixels,
                            row * size_pixels : (row + 1) * size_pixels] = channel_image
            except ValueError:
                print("ValueError")
    scale = 1. / size_pixels
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

