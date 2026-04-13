'''tf_util.py
Helper/utility functions related to using TensorFlow for Transfer Learning and working with images
Dean Hickman
CS 343: Neural Networks
Project 4: Transfer Learning
Spring 2025
'''
import numpy as np
from PIL import Image
import tensorflow as tf


def load_pretrained_net(net_name='vgg19'):
    '''Loads the pretrained network (included in Keras) identified by the string `net_name`.

    Parameters:
    -----------
    net_name: str. Name of pretrained network to load. By default, this is VGG19.

    Returns:
    -----------
    The pretained net. Keras object.

    NOTE: Pretrained net should NOT be trainable and NOT include the output layer.
    '''
    if net_name == 'vgg19':
        net = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
    elif net_name == 'resnet50':
        net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    elif net_name == 'inceptionv3':
        net = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
    elif net_name == 'mobilenetv2':
        net = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    elif net_name == 'densenet201':
        net = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
    else:
        print('Screenshot 2024-11-01 at 4.20.06 PM.png')
    net.trainable = False 

    return net


def get_all_layer_strs(pretrained_net):
    '''Gets the complete list of layer names from the pretrained net.

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network.

    Returns:
    -----------
    Python list of str. Length is the number of layers in the pretrained network.
    '''
    layer_names = []
    for layer in pretrained_net.layers:
        layer_names.append(layer.name)
    return layer_names

def filter_layer_strs(layer_names, match_str='conv4'):
    '''Extracts the layer name strs from `layer_names` (the complete list) that have `match_str` in the name.

    Parameters:
    -----------
    layer_names: Python list of str. The complete list of layer names in the pretrained network
    match_str: str. Substring searched for within each layer name

    Returns:
    -----------
    Python list of str. The list of layers from `layer_names` that include the string `match_str`
    '''
    return [layer_name for layer_name in layer_names if match_str in layer_name]


def preprocess_image2tf(img, as_var):
    '''Converts an image (in numpy ndarray format) to TensorFlow tensor format

    Parameters:
    -----------
    img: ndarray. shape=(Iy, Ix, n_chans). A single image
    as_var: bool. Do we represent the tensor as a tf.Variable?

    Returns:
    -----------
    tf tensor. dtype: tf.float32. shape=(1, Iy, Ix, n_chans)

    NOTE: Notice the addition of the leading singleton batch dimension in the tf tensor returned.
    '''
    img = np.expand_dims(img, axis=0).astype(np.float32)
    if np.max(img) > 1.0:
        img = img / 255.0
    if as_var:
        return tf.Variable(img)
    else:
        return tf.convert_to_tensor(img)


def make_readout_model(pretrained_net, layer_names):
    '''Makes a tf.keras.Model object that returns the netAct (output) values of layers in the pretrained model
    `pretrained_net` that have the names in the list `layer_names` (the readout model).

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network
    layer_names: Python list of str. Selected list of pretrained net layer names whose netAct values should be returned
        by the readout model.

    Returns:
    -----------
    tf.keras.Model object (readout model) that provides a readout of the netAct values in the selected layer list
        (`layer_names`).
    '''
    inputs = pretrained_net.input
    outputs = [pretrained_net.get_layer(layer_name).output for layer_name in layer_names]
    readout_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return readout_model


def tf2image(tensor):
    '''Converts a TensorFlow tensor into a PIL Image object.

    Parameters:
    -----------
    tensor: tf tensor. dtype=tf.float32. shape=(1, Iy, Ix, n_chans). A single image. Values range from 0-1.

    Returns:
    -----------
    PIL Image object. dtype=uint8. shape=(Iy, Ix, n_chans). Image representation of the input tensor with pixel values
        between 0 and 255 (unsigned ints).

    NOTE:
    - Scale pixel values to the range [0, 255] BEFORE converting to uint8 dtype.
    - Remove batch (singleton) dimension (if present)
    - One way to convert to PIL Image is to first convert to numpy ndarray.

    The following should be helpful:
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    '''
    tensor = tf.squeeze(tensor, axis=0)
    tensor = tf.cast(tensor * 255, tf.uint8)
    img_array = tensor.numpy()
    img = Image.fromarray(img_array)
    return img
