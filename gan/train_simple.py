import os
import time
import numpy as np
import keras
from keras import layers
from keras.preprocessing import image

latent_dim = 32
height = 32
width = 32
channels = 3

def get_generator_model():
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    return generator

def get_discriminator_model():
    
    discriminator_input = layers.Input(shape=(height, width, channels))
    
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)

    return discriminator

def get_gan(generator, discriminator):

    discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    return gan, discriminator

def run_model(gan, generator, discriminator, training_images, num_iterations, dir_output, batch_size=20):

    start = 0

    print('step\tdiscriminator loss\tadversial loss\telapsed time:')

    for step in range(1, num_iterations+1):

        time_start = time.time()

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)

        stop = start + batch_size

        real_images = training_images[start:stop]
        combined_images = np.concatenate([generated_images, real_images])        

        labels = np.concatenate([np.ones((batch_size, 1)), 
                                        np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        misleading_targets = np.zeros((batch_size, 1))

        # train the generator through the gan model, the discriminator weights are frozen
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        start += batch_size

        if start > len(training_images) - batch_size:
            start = 0

        time_end = time.time()
        time_elapsed = time_end - time_start

        #if step % 10 == 0:
        print(step, '\t', d_loss, '\t', a_loss, '\t', time_elapsed)

        #if step % 100 == 0:
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(dir_output, 'generated_' + str(step) + '.png'))
    
def main():

    # 1. prepare data for model

    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = x_train[y_train.flatten() == 1]
    x_train = x_train.reshape(((x_train.shape[0],) + (height, width, channels))).astype('float32') / 255.

    # 2. create the model    

    generator = get_generator_model()
    generator.summary()

    discriminator = get_discriminator_model()
    discriminator.summary()

    gan, discriminator = get_gan(generator, discriminator)
    gan.summary()

    # 3. train for num_iterations

    num_iterations = 10
    batch_size = 20
    dir_output = 'images'

    run_model(gan, generator, discriminator, x_train, num_iterations, dir_output, batch_size=batch_size)

if __name__ == "__main__":
    main()
