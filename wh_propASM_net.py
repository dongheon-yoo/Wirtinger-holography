"""
2020. 01. 04 
Follow up code for
P. Chakravarthula et al., "Wirtinger Holography for Near-Eye Displays", 
SIGGRAPH ASIA, ACM TOG, 2019

by Dongheon Yoo

Optimizing phase hologram via gradient descent algorithm
Use Tensorflow 2.0
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import wh_propASM_config as config
import wh_propASM_util as util

PARAM = config.PARAM

class Interleave(layers.Layer):
    def __init__(self, rate, name = 'interleave', **kwargs):
        super(Interleave, self).__init__()
        self.rate = rate # interleaving rate
        self.l_name = name

    def call(self, input_tensor):
        return tf.compat.v1.space_to_depth(input_tensor, self.rate, name = self.l_name)

class Deinterleave(layers.Layer):
    def __init__(self, rate, name = 'deinterleave', **kwargs):
        super(Deinterleave, self).__init__()
        self.rate = rate # Deinterleaving rate, which MUST BE EQUAL to the interleaving rate.
        self.l_name = name

    def call(self, input_tensor):
        return tf.compat.v1.depth_to_space(input_tensor, self.rate, name = self.l_name)

class ASM2_propagation(layers.Layer):
    def __init__(self,
                 prop_distance,
                 pixel_pitch,
                 wavelength,
                 **kwargs):
        super(ASM2_propagation, self).__init__()
        global PARAM
        self.slm_h, self.slm_w = PARAM.get('SLM_HEIGHT'), PARAM.get('SLM_WIDTH')
        self.z = prop_distance
        self.pp = pixel_pitch
        self.wl = wavelength

        # Set zero padding size
        self.padH = int(np.ceil(self.slm_h / 2))
        self.padW = int(np.ceil(self.slm_w / 2))

        # Set Spatial Frequency Domain
        T = 1 / self.pp
        oh, ow = (self.slm_h + 2 * self.padH), (self.slm_w + 2 * self.padW)
        dfy, dfx = T / oh, T / ow
        self.fx_grid, self.fy_grid = util.square_grid([oh, ow], dfy)

    def call(self, field):
        # Zero pad the input field
        field_pad = tf.pad(field, [[0, 0], [self.padH, self.padH], [self.padW, self.padW], [0, 0]])
        gamma = ((1 / self.wl) ** 2) - self.fx_grid ** 2 - self.fy_grid ** 2
        gamma = gamma[np.newaxis, :, :, np.newaxis]
        mask = np.array(gamma > 0).astype(np.float32)
        gamma = mask * gamma
        AS = util.FT_tf(field_pad)
        TF = np.exp(1.j * 2 * np.pi * np.sqrt(gamma) * self.z)
        ASTF = AS * tf.convert_to_tensor(TF, dtype = tf.complex64)
        out_field = util.iFT_tf(ASTF)
        out_field = tf.slice(out_field, [0, self.padH, self.padW, 0], [-1, self.slm_h, self.slm_w, -1])

        return out_field

def encode_double_phase(field):
    # field : complex wavefront
    phase_t = tf.math.angle(field)
    amp_t = tf.math.abs(field) / tf.reduce_max(tf.math.abs(field), axis = [1, 2], keepdims = True)
    pa = phase_t - tf.math.acos(amp_t)
    pb = phase_t + tf.math.acos(amp_t)
    pa_int = Interleave(2)(pa)
    pb_int = Interleave(2)(pb)

    # Constant
    pa_slice0 = tf.slice(pa_int, [0, 0, 0, 0], [-1, -1, -1, 1])
    pa_slice3 = tf.slice(pa_int, [0, 0, 0, 3], [-1, -1, -1, 1])
    pb_slice1 = tf.slice(pb_int, [0, 0, 0, 1], [-1, -1, -1, 1])
    pb_slice2 = tf.slice(pb_int, [0, 0, 0, 2], [-1, -1, -1, 1])
    phi_temp = tf.concat([pa_slice0, pb_slice1, pb_slice2, pa_slice3], axis = -1)
    phi = Deinterleave(2)(phi_temp)

    return phi
