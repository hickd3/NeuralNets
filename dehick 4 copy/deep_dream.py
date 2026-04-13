'''deep_dream.py
Generate art with a pretrained neural network using the DeepDream algorithm
Dean Hickman
CS 343: Neural Networks
Project 4: Transfer Learning
Spring 2025
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

import tf_util


class DeepDream:
    '''Runs the DeepDream algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, selected_layers_names):
        '''DeepDream constructor.

        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        selected_layers_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values
            from in order to contribute to the generated image.

        TODO:
        1. Define instance variables for the pretrained network and the number of selected layers used to readout netAct.
        2. Make an readout model for the selected layers (use function in `tf_util`) and assign it as an instance variable.
        '''
        self.loss_history = None
        self.pretrained_net = pretrained_net
        self.num_selected_layers = len(selected_layers_names)
        self.readout_model = tf_util.make_readout_model(pretrained_net, selected_layers_names)

    def loss_layer(self, layer_net_acts):
        '''Computes the contribution to the total loss from the current layer with netAct values `layer_net_acts`. The
        loss contribution is the mean of all the netAct values in the current layer.

        Parameters:
        -----------
        layer_net_acts: tf tensor. shape=(1, Iy, Ix, K). The netAct values in the current selected layer. K is the
            number of kernels in the layer.

        Returns:
        -----------
        loss component from current layer. float. Mean of all the netAct values in the current layer.
        '''
        return tf.reduce_mean(layer_net_acts)
    
    def forward(self, gen_img, standardize_grads=True, eps=1e-8):
        '''Performs forward pass through the pretrained network with the generated image `gen_img`.
        Loss is computed based on the SELECTED layers (in readout model).

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. The singleton dimension is the batch dimension (N).
        standardize_grads: bool. Should we standardize the image gradients?
        eps: float. Small number used in standardization to prevent possible division by 0 (i.e. if stdev is 0).

        Returns:
        -----------
        loss. float. Sum of the loss components from all the selected layers.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.

        TODO:
        While tracking gradients:
        - Use the readout model to extract the netAct values in the selected layers for `gen_img`.
        - Compute the average loss across all selected layers.
        Then:
        - Obtain the tracked gradients of the loss with respect to the generated image.
        '''
        with tf.GradientTape() as tape:
            tape.watch(gen_img)
            layer_net_acts = self.readout_model(gen_img)
            loss = tf.reduce_mean([self.loss_layer(act) for act in layer_net_acts])
            grads = tape.gradient(loss, gen_img)
            if standardize_grads:
                mu = tf.reduce_mean(grads)
                sigma = tf.math.reduce_std(grads)
                grads = (grads - mu) / (sigma + eps)
        return loss, grads

    def fit(self, gen_img, n_epochs=26, lr=0.01, print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''Iteratively modify the generated image (`gen_img`) for `n_epochs` with the image gradients using the
            gradient ASCENT algorithm. In other words, run DeepDream on the generated image.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one. For example, image_1.jpg, image_2.jpg,
            etc.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Compute the forward pass on the generated image for `n_epochs`.
        2. Apply the gradients to the generated image using the gradient ASCENT update rule.
        3. Clip pixel values to the range [0, 1] and update the generated image.
            The TensorFlow `assign` function is helpful here because = would "wipe out" the tf.Variable property,
            which is not what we want because we want to track gradients on the generated image across epochs.
        4. After the first epoch completes, always print out how long it took to finish the first epoch and an estimate
        of how long it will take to complete all the epochs (in minutes).

        NOTE:
        - Deep Dream performs gradient ASCENT rather than DESCENT (which we are more used to). The difference is only
        in the sign of the gradients.
        - Clipping is different than normalization!
        '''
        self.loss_history = []
        start_time = time.time()
        first_epoch_time = None

        for epoch in range(n_epochs):
            epoch_start = time.time()
            loss, grads = self.forward(gen_img)
            gen_img.assign(tf.clip_by_value(gen_img + lr * grads, 0, 1))
            epoch_time = time.time() - epoch_start
            self.loss_history.append(loss.numpy())

            if epoch % print_every == 0 or epoch == 0:
                if first_epoch_time is None:
                    first_epoch_time = epoch_time
                    estimated_total = first_epoch_time * n_epochs
                    print(f"Epoch {epoch}/{n_epochs-1} completed")
                    print(f'Estimated time to complete all epochs: {estimated_total / 60:.2f} minutes')

                if plot:
                    img = tf_util.tf2image(gen_img)
                    plt.figure(figsize=plot_fig_sz)
                    plt.imshow(img)
                    plt.title(f"Epoch {epoch + 1}")
                    plt.axis("off")
                    plt.show()

                if export:
                    output_dir = 'deep_dream_output'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    filename = f'image_{epoch + 1}.jpg'
                    img = tf_util.tf2image(gen_img)
                    img.save(os.path.join(output_dir, filename))

        return self.loss_history

    def fit_multiscale(self, gen_img, n_scales=4, scale_factor=1.3, n_epochs=26, lr=0.01, print_every=1, plot=True,
                       plot_fig_sz=(5, 5), export=True):
        '''Run DeepDream `fit` on the generated image `gen_img` a total of `n_scales` times. After each time, scale the
        width and height of the generated image by a factor of `scale_factor` (round to nearest whole number of pixels).
        The generated image does NOT start out from scratch / the original image after each resizing. Any modifications
        DO carry over across runs.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_scales: int. Number of times the generated image should be resized and DeepDream should be run.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current scale) every this many SCALES (not epochs).
        plot: bool. If true, plot/show the generated image `print_every` SCALES.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` SCALES. Each exported image should have the current scale number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Call fit `n_scale` times. Pass along hyperparameters (n_epochs, etc.). Turn OFF plotting and exporting within
        the `fit` method — this method should take over the plotting and exporting (in scale intervals rather than epochs).
        2. Multiplicatively scale the generated image.
            Check out: https://www.tensorflow.org/api_docs/python/tf/image/resize

            NOTE: The output of the built-in resizing process is NOT a tf.Variable (its an ordinary tensor).
            But we need a tf.Variable to compute the image gradient during gradient ascent.
            So, wrap the resized image in a tf.Variable.
        3. After the first scale completes, always print out how long it took to finish the first scale and an estimate
        of how long it will take to complete all the scales (in minutes).
        '''
        self.loss_history = []
        start_time = time.time()
        first_scale_time = None

        for scale in range(n_scales):
            if scale > 0:
                new_height = int(gen_img.shape[1] * scale_factor)
                new_width = int(gen_img.shape[2] * scale_factor)
                gen_img = tf.image.resize(gen_img[0], (new_height, new_width))
                gen_img = tf.Variable(gen_img[None, ...])  # Wrap resized image in a tf.Variable

            for epoch in range(n_epochs):
                epoch_start = time.time()
                loss, grads = self.forward(gen_img)
                gen_img.assign(tf.clip_by_value(gen_img + lr * grads, 0, 1))
                epoch_time = time.time() - epoch_start
                self.loss_history.append(loss.numpy())

                if epoch % print_every == 0 or epoch == 0:
                    if first_scale_time is None:
                        first_scale_time = epoch_time
                        estimated_total = first_scale_time * n_scales * n_epochs
                        print(f"Scale {scale + 1}/{n_scales} completed")
                        print(f'Estimated time to complete all scales: {estimated_total / 60:.2f} minutes')

                    if plot:
                        img = tf_util.tf2image(gen_img)
                        plt.figure(figsize=plot_fig_sz)
                        plt.imshow(img)
                        plt.title(f"Scale {scale + 1}, Epoch {epoch + 1}")
                        plt.axis("off")
                        plt.show()

                    if export:
                        output_dir = 'deep_dream_output'
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        filename = f'scale_{scale + 1}_epoch_{epoch + 1}.jpg'
                        img = tf_util.tf2image(gen_img)
                        img.save(os.path.join(output_dir, filename))

        return self.loss_history
