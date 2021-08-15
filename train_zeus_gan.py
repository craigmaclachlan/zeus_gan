#!/usr/bin/env python3

import numpy
import os
import sys
import time
import argparse
import glob

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

EPOCHS = 3000
DIAG_UPDATES = 1
MODEL_CHKPT = 1
NOISE_DIM = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 12
NUM_EXAMPLES_TO_GENERATE = 5

# LR_D = 0.00005
# LR_G = 0.0005
LR_D = 0.00002
LR_G = 0.0005


BETA1 = 0.5
#MINI_BATCH = 40
RANDOM_SEED = 5
EPSILON = 0.00005
WEIGHT_INIT_STDDEV = 0.02

def make_generator_model_r128():
    #initializer = tf.keras.initializers.TruncatedNormal
    initializer = tf.keras.initializers.RandomNormal

    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 1024, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 1024)))

    # 8x8x1024 -> 16x16x512
    model.add(
        layers.Conv2DTranspose(
            512,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
            use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 16x16x512 -> 32x32x256
    model.add(
        layers.Conv2DTranspose(
            256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
            use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 32x32x256 -> 64x64x128
    model.add(layers.Conv2DTranspose(
        128,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
        use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 64x64x128 -> 128x128x64
    model.add(layers.Conv2DTranspose(
        64,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
        use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 128x128x64 -> 128x128x3
    model.add(layers.Conv2DTranspose(
        3,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                       seed=numpy.random.randint(99999)),
        use_bias=False,
        activation='tanh')
    )
    print(model.output_shape)
    assert model.output_shape == (None, 128, 128, 3)

    return model

def make_generator_model_r256():

    #initializer = tf.keras.initializers.TruncatedNormal
    initializer = tf.keras.initializers.RandomNormal

    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 1024, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 1024)))

    # 8x8x1024 -> 16x16x512
    model.add(
        layers.Conv2DTranspose(
            512,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
            use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 16x16x512 -> 32x32x256
    model.add(
        layers.Conv2DTranspose(
            256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                           seed=numpy.random.randint(99999)),
            use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 32x32x256 -> 64x64x128
    model.add(layers.Conv2DTranspose(
        128,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                       seed=numpy.random.randint(99999)),
        use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 64x64x128 -> 128x128x64
    model.add(layers.Conv2DTranspose(
        64,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                       seed=numpy.random.randint(99999)),
        use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 128x128x64 -> 256x256x32
    model.add(layers.Conv2DTranspose(
        32,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                       seed=numpy.random.randint(99999)),
        use_bias=False)
    )
    print(model.output_shape)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())

    # 256x256x32 -> 256x256x3
    model.add(layers.Conv2DTranspose(
        3,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer(stddev=WEIGHT_INIT_STDDEV,
                                       seed=numpy.random.randint(99999)),
        use_bias=False,
        activation='tanh')
    )
    print(model.output_shape)
    assert model.output_shape == (None, 256, 256, 3)

    return model


def make_discriminator_model_r128():

    #initializer = tf.keras.initializers.TruncatedNormal
    initializer = tf.keras.initializers.RandomNormal

    model = tf.keras.Sequential()

    # 128*128*3 -> 128x128x64
    model.add(layers.Conv2D(64, (4, 4),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            input_shape=[128, 128, 3])
              )
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 128*128*64 -> 64x64x128
    model.add(layers.Conv2D(128, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same')
              )
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 64x64x128-> 32x32x256
    model.add(layers.Conv2D(256, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 32x32x256 -> 16x16x512
    model.add(layers.Conv2D(512, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 16x16x512 -> 8x8x1024
    model.add(layers.Conv2D(1024, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    #model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.Dense(1))

    return model


def make_discriminator_model_r256():

    #initializer = tf.keras.initializers.TruncatedNormal
    initializer = tf.keras.initializers.RandomNormal

    model = tf.keras.Sequential()

    # 256*256*3 -> 256x256x32
    model.add(layers.Conv2D(32, (4, 4),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            input_shape=[256, 256, 3])
              )
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 256*256*32 -> 128x128x64
    model.add(layers.Conv2D(64, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same')
              )
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 128x128x64 -> 64x64x128
    model.add(layers.Conv2D(128, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same')
              )
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 64x64x128-> 32x32x256
    model.add(layers.Conv2D(256, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 32x32x256 -> 16x16x512
    model.add(layers.Conv2D(512, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    # 16x16x512 -> 8x8x1024
    model.add(layers.Conv2D(1024, (4, 4),
                            strides=(2, 2),
                            kernel_initializer=initializer(
                                stddev=WEIGHT_INIT_STDDEV,
                                seed=numpy.random.randint(99999)),
                            padding='same'))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    #model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.Dense(1))

    return model


def smooth_positive_labels(y):
	return y - 0.2 + (numpy.random.random(y.shape) * 0.2)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        smooth_positive_labels(tf.ones_like(real_output)),
        real_output
    )
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = tf.math.reduce_mean(real_loss + fake_loss)
    return total_loss


def generator_loss(fake_output):
    return tf.math.reduce_mean(
        cross_entropy(tf.ones_like(fake_output), fake_output)
    )


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # TODO: consider adding noise to the true images?
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input, img_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(12, 3))
    for i in range(NUM_EXAMPLES_TO_GENERATE):
        plt.subplot(1, NUM_EXAMPLES_TO_GENERATE, i+1)
        img = numpy.clip((predictions[i, :, :, :].numpy()+1.0)/2.0, 0.0, 1.0)
        plt.imshow(img)
        plt.axis('off')

    img_path = os.path.join(img_dir, 'image_at_epoch_{:06d}.png'.format(epoch))
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_losses(epoch, d_losses, g_losses, plot_dir):
    """Plot the loss values from the generator and discriminator."""

    ep_range = range(1, epoch+1)[-len(d_losses):]
    fig = plt.figure()
    if len(d_losses) < 200:
        plt.plot(ep_range, d_losses, label='Discriminator', alpha=0.6)
        plt.plot(ep_range, g_losses, label='Generator', alpha=0.6)
    else:
        plt.plot(ep_range[-200:], d_losses[-200:], label='Discriminator', alpha=0.6)
        plt.plot(ep_range[-200:], g_losses[-200:], label='Generator', alpha=0.6)

    plt.xlabel("Epoch")
    plt.title("Losses")
    plt.legend()
    loss_img = os.path.join(plot_dir, "losses.png")
    plt.savefig(loss_img)
    plt.close(fig)


def train(dataset, epochs, test_img_seeds, checkpoint, cpoint_manager, img_dir,
          n_mini_batch):
    print("Starting to train.")
    d_losses = []
    g_losses = []

    for epoch in range(int(checkpoint.step), epochs):
        start = time.time()
        gen_loss_avg = tf.keras.metrics.Mean()
        dis_loss_avg = tf.keras.metrics.Mean()

        for ii in range(n_mini_batch):
            image_batch = next(dataset)
            gloss, dloss = train_step(image_batch)
            dis_loss_avg(dloss)
            gen_loss_avg(gloss)
            #d_losses.append(dloss)
            #g_losses.append(gloss)

        d_losses.append(dis_loss_avg.result())
        g_losses.append(gen_loss_avg.result())

        print("Epoch {}/{}".format(epoch, epochs),
              "\nD Loss: {:.5f}".format(d_losses[-1]),
              "\nG Loss: {:.5f}".format(g_losses[-1]))

        # Produce images for the GIF as we go
        if epoch % DIAG_UPDATES == 0:
            generate_and_save_images(generator, epoch, test_img_seeds,
                                     img_dir)
            plot_losses(epoch, d_losses, g_losses, img_dir)

        # Save the model every epoch
        if int(checkpoint.step) % MODEL_CHKPT == 0:
            save_path = cpoint_manager.save()
            print(
                "Saved checkpoint for step {}: {}".format(int(checkpoint.step),
                                                          save_path)
            )
        checkpoint.step.assign_add(1)

        print('Time for epoch {} is {} sec'.format(epoch,
                                                   time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, test_img_seeds, img_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train GAN to generate Zeuses')
    parser.add_argument('res', help='Resolution of images: 128 or 256.',
                        type=int)
    parser.add_argument('training_images',
                        help='Path to directory containing training images.'
                             'Images need to be separated into sub-directories'
                             'with the class name (even though these are ignored.')
    parser.add_argument('output_root',
                        help='Path to directory that will contain all the '
                             'output: model checkpoints, test images, etc.')

    args = parser.parse_args()

    n_images = len(glob.glob(args.training_images + '/*/*'))
    n_mini_batch = n_images // BATCH_SIZE
    n_mini_batch = n_mini_batch if n_mini_batch > 200 else 200
    checkpoint_dir = os.path.join(args.output_root, 'checkpoint')
    gen_img_dir = os.path.join(args.output_root, 'images')
    print('Looking for training images in: %s' % args.training_images)
    print('Number of training images = %d' % n_images)
    print('Number of mini batches = %d' % n_mini_batch)
    print('Model output will be stored: %s' % args.output_root)
    print('Model checkpoints will be stored: %s' % checkpoint_dir)
    print('Generated test images will be stored: %s' % gen_img_dir)

    try:
        os.makedirs(checkpoint_dir)
        os.makedirs(gen_img_dir)
        print("Model directories created.")
    except FileExistsError:
        print("Model directories exist.")
        pass

    image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=25,
        #width_shift_range=0.15,
        #height_shift_range=0.15,
        fill_mode='nearest',
        preprocessing_function=lambda x: x/127.5 - 1.0)

    train_data_gen = image_gen_train.flow_from_directory(
        args.training_images,
        target_size=(args.res, args.res),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=True)

    if args.res == 128:
        generator = make_generator_model_r128()
        discriminator = make_discriminator_model_r128()
    elif args.res == 256:
        generator = make_generator_model_r256()
        discriminator = make_discriminator_model_r256()
    else:
        print("ERROR: Invalid resolution: %s" % args.res)
        sys.exit()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_G,
                                                   beta_1=BETA1)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_D,
                                                       beta_1=BETA1)

    # Set up the model checkpointing
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(1),
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=3)

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    random_normal = tf.random_normal_initializer(seed=RANDOM_SEED, stddev=1.0)
    test_seeds = random_normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    train(train_data_gen, EPOCHS, test_seeds, checkpoint, manager, gen_img_dir,
          n_mini_batch)
