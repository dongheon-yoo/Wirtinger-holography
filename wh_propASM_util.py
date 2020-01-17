"""
2020. 01. 04 
Follow up code for
P. Chakravarthula et al., "Wirtinger Holography for Near-Eye Displays", 
SIGGRAPH ASIA, ACM TOG, 2019

by Dongheon Yoo
Optical Engineering and Quantum Electronics Laboratory,
Seoul National University, Korea.

Optimizing phase hologram via gradient descent algorithm
Use Tensorflow 2.0
"""

import os
import numpy as np
import math
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wh_propASM_config as config

PARAM = config.PARAM

def make_dir(path, overwrite=False):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.exists(path) and overwrite:
        os.remove(path)
        os.mkdir(path)

def load_and_preprocess_train(train_path):
    global PARAM

    # Read RGB clean image
    rgb_name = train_path.numpy().decode("utf-8")
    slm_h, slm_w = PARAM.get('SLM_HEIGHT'), PARAM.get('SLM_WIDTH')
    wl = PARAM.get('WAVELENGTH')
    pp = PARAM.get('PIXEL_PITCH')
    z = PARAM.get('PROPAGATION_DISTANCE')

    # Read single color image
    im_gray = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
    im_gray = im_gray[:, :, [0]]
    im_gray = cv2.normalize(im_gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    height, width = im_gray.shape
    if height != width:
        im_size = np.minimum(height, width)
        im_gray = im_gray[0 : im_size, 0 : im_size]

    # Pre-pad array
    theta = np.arcsin(wl / (2 * pp))
    padSize = np.abs(z) * np.tan(theta)
    padN = int(np.ceil(padSize / pp))
    oh, ow = (slm_h - 2 * padN), (slm_w - 2 * padN)
    im_gray = cv2.resize(im_gray, (oh, ow))
    im_gray = np.clip(im_gray, 0.0, 1.0)
    im_gray = np.pad(im_gray, ((padN, padN), (padN, padN)), mode = 'constant')
    im_gray = im_gray[:, :, np.newaxis]

    return im_gray

def parse_py_func_train(train_path):
    global PARAM

    slm_h, slm_w = PARAM.get('SLM_HEIGHT'), PARAM.get('SLM_WIDTH')

    out_func = tf.py_function(load_and_preprocess_train, [train_path], [tf.float32])
    out_func[0].set_shape(tf.TensorShape((slm_h, slm_w, 1)))

    return out_func

def parse_dict_train(im_gray):
    global PARAM
    return {'im_gray' : im_gray}

def transpose_fft2d(a_tensor, name = None, dtype = tf.complex64):
    """
    Operates Tensorflow's 2D FFT on transposed images
    of shape [batch_size, channels, height, width]
    as tf.fft2d operates on the two innermost (last two) dimensions.

    Input parameters:
    a_tensor : [batch_size, height, width, channels]

    Output:
    a_fft2d : [batch_size, height, width, channels]
    """
    a_tensor = tf.cast(a_tensor, dtype)
    a_tensor_T = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.cast(tf.signal.fft2d(a_tensor_T), dtype = dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1], name = name)
    return a_fft2d

def transpose_ifft2d(a_tensor, name = None, dtype = tf.complex64):
    """
    Operates Tensorflow's 2D iFFT on transposed images
    of shape [batch_size, channels, height, width]
    as tf.ifft2d operates on the two inner-most (last two) dimensions.

    Input parameters:
    a_tensor : [batch_size, height, width, channels]

    Output:
    a_fft2d : [batch_size, height, width, channels]
    """
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_tensor_T = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_ifft2d = tf.cast(tf.signal.ifft2d(a_tensor_T), dtype = dtype)
    a_ifft2d = tf.transpose(a_ifft2d, [0, 2, 3, 1], name = name)
    return a_ifft2d

def FT_tf(a_tensor, name = None, dtype = tf.complex64):
    new_tensor = tf.signal.ifftshift(a_tensor, axes = [1, 2])
    new_tensor = transpose_fft2d(new_tensor, name, dtype)
    new_tensor = tf.signal.fftshift(new_tensor, axes = [1, 2])
    return new_tensor

def iFT_tf(a_tensor, name = None, dtype = tf.complex64):
    new_tensor = tf.signal.ifftshift(a_tensor, axes = [1, 2])
    new_tensor = transpose_ifft2d(new_tensor, name, dtype)
    new_tensor = tf.signal.fftshift(new_tensor, axes = [1, 2])
    return new_tensor

def complex_exp_tf(phase, name = None, dtype = tf.complex64):
    """
    Complex exponent via euler's formula.
    """
    phase = tf.cast(phase, dtype = tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype = dtype),
                  1j * tf.cast(tf.sin(phase), dtype = dtype), name)

def square_grid(resolution, physical_pitch, dtype = np.float32):
    """
    Generates square grid using meshgrid.

    Input paramters:
    resolution : shape of [2] (height, width)
    physical_pitch : shape of [2] (pitch_y, pitch_x)

    Output parameters:
    x_grid : shape of [1, height, width, 1]
    y_grid : shape of [1, height, width, 1]
    """
    height, width = resolution
    pitch_y = physical_pitch
    pitch_x = pitch_y
    x_vector = np.linspace(- width / 2, width / 2, num = width, endpoint = False) * pitch_x
    y_vector = np.linspace(- height / 2, height / 2, num = height, endpoint = False) * pitch_y
    x_grid, y_grid = np.meshgrid(x_vector.astype(dtype), y_vector.astype(dtype))

    return x_grid, y_grid
