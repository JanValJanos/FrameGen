from __future__ import print_function, division
from DataLoader import DataLoader
import scipy

# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_distance_transform(image):
    y = tf.numpy_function(distance_transform_float, [image], tf.float32)
    return y


class FrameNetwork():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'facades'
        # Comentando pois não precisa treinar aqui
        #self.data_loader = AsdDataLoader(dataset_name=self.dataset_name,
        #                                 img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)
        # optimizer = Adam(0.00002, 0.5) Abaixar learning rate

        # Build and compile the discriminators
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)

        # By conditioning on A and C generate a fake version of B
        fake_B = self.generator([img_A, img_C])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_A, fake_B, img_C])

        self.combined = Model(inputs=[img_A, img_B, img_C], outputs=[valid, fake_B])
        self.combined.compile(loss=['binary_crossentropy', 'mae'],  # 'mse'],#'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

        # tf.compat.v1.enable_eager_execution()

    def custom_loss(self, y_actual, y_predicted):
        image_loss = mean_squared_error(y_actual, y_predicted)

        # Comentado pois ainda não funcionou
        dist_actual = tf_distance_transform(y_actual)
        dist_predicted = tf_distance_transform(y_predicted)
        dist_loss = mean_squared_error(dist_actual, dist_predicted)

        loss = tf.math.divide(tf.math.add(tf.math.multiply(image_loss, tf.constant([2], dtype=tf.float32)), dist_loss),
                              tf.constant([3], dtype=tf.float32))
        # loss = dist_loss
        # loss = image_loss

        return loss

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        img_A = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)

        # Concatenate conditioning images
        combined_imgs = Concatenate(axis=-1)([img_A, img_C])

        # Downsampling
        d1 = conv2d(combined_imgs, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model([img_A, img_C], output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True, n_strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=n_strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B, img_C])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)
        d5 = d_layer(d4, self.df * 8, n_strides=1)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        activation = Activation('sigmoid')(validity)

        return Model([img_A, img_B, img_C], activation)

    def train(self, epochs, batch_size=1, sample_interval=50, patience=5, early_stopping=False):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        losses = {}
        remaining_patience = patience

        for epoch in range(epochs):
            epoch_losses = []
            for batch_i, (imgs_A, imgs_B, imgs_C) in enumerate(self.data_loader.load_batch(batch_size, augment=True)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_B = self.generator.predict([imgs_A, imgs_C])

                # Train the discriminators (original images = real / generated = Fake)
                # d_loss_real_A = self.discriminator_douga.train_on_batch(imgs_A, valid)
                # d_loss_real_B = self.discriminator_douga.train_on_batch(imgs_B, valid)
                # d_loss_real_C = self.discriminator_douga.train_on_batch(imgs_C, valid)
                # d_loss_real = (np.asarray(d_loss_real_A) + np.asarray(d_loss_real_B) + np.asarray(d_loss_real_C)) / 3

                # Comentado para deixar o discriminador mais fraco, pois estava alcançando erro 0 muito rápido (< 500 epochs)

                # Train the discriminators (original images = real / generated = Fake)
                if batch_i % 4 == 0:
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B, imgs_C], valid)
                    d_loss_fake = self.discriminator.train_on_batch([imgs_A, fake_B, imgs_C], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # if batch_i % 10 == 0:
                    #     # Train the discriminators (original images = real / generated = Fake)
                    #     d_loss_real = self.discriminator.train_on_batch([imgs_B, imgs_B, imgs_B, imgs_flow], valid)
                    #     d_loss_fake = self.discriminator.train_on_batch([imgs_B, fake_B, imgs_B, imgs_flow], fake)
                    #     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, imgs_C], [valid, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      self.data_loader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                epoch_losses.append(g_loss[0])

                ## Segundo treino na mesma epoch
                # fake_B = self.generator.predict([imgs_C, imgs_Cd, imgs_A, imgs_Ad])
                # g_loss = self.combined.train_on_batch([imgs_C, imgs_Cd, imgs_B, imgs_Bd, imgs_A, imgs_Ad], [valid, valid, imgs_B])

                # if batch_i % 3 == 0:
                #    fake_B = self.generator.predict([imgs_B, imgs_Bd, imgs_B, imgs_Bd])
                #    g_loss = self.combined.train_on_batch([imgs_B, imgs_Bd, imgs_B, imgs_Bd, imgs_B, imgs_Bd], [valid, valid, imgs_B])

                # If at save interval => save generated image samples
                # if batch_i % sample_interval == 0:
                #    self.sample_images(epoch, batch_i)

            losses[epoch] = np.average(epoch_losses)

            if len(losses) > 1 and early_stopping:
                if losses[list(losses.keys())[-1]] > losses[list(losses.keys())[-2]]:
                    remaining_patience -= 1
                    print("Remaining patience: %d" % remaining_patience)
                    if remaining_patience == 0:
                        break
                else:
                    remaining_patience = patience

        return losses

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B, imgs_C = self.data_loader.load_data(batch_size=3, ids=["CRL000", "SQR008",
                                                                               "SQR088"])  # ids=["a00-c00", "a01-c01", "a15-c04"])#["s00-c00", "s01-c03", "s01-c22"])#, is_testing=True)
        fake_B = self.generator.predict([imgs_A, imgs_C])

        gen_imgs = np.concatenate([[cv2.cvtColor(imA, cv2.COLOR_BGR2RGB) for imA in imgs_A],
                                   [cv2.cvtColor(fakeB, cv2.COLOR_BGR2RGB) for fakeB in fake_B],
                                   [cv2.cvtColor(imB, cv2.COLOR_BGR2RGB) for imB in imgs_B]])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def load_weights(self, filepath):
        self.combined.load_weights(filepath)