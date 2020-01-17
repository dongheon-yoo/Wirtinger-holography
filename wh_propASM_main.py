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
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import wh_propASM_config as config
import wh_propASM_net as Net
import wh_propASM_util as util

# Network parameter
PARAM = config.PARAM
np.random.seed(888)
# List all GPUs, Allocate only needed memory for process
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    tf.random.set_seed(777)

    # Basic parameters
    lr = PARAM.get('LEARNING_RATE')
    result_dir = PARAM.get('RESULT_DIR')
    batch_size = PARAM.get('BATCH_SIZE')
    n_epoch = PARAM.get('EPOCH')
    steps_per_plot_image = PARAM.get('STEPS_PER_PLOT_IMAGE')

    slm_h, slm_w = PARAM.get('SLM_HEIGHT'), PARAM.get('SLM_WIDTH')
    pp = PARAM.get('PIXEL_PITCH')
    wavelength = PARAM.get('WAVELENGTH')
    z_prop = PARAM.get('PROPAGATION_DISTANCE')

    # Dataset loader (for fast GPU image loading)
    # Target image should be located in current folder
    image_path = glob.glob(os.path.join(os.getcwd(), '0076.png'))
    train_dataset = Dataset.from_tensor_slices(image_path)

    # Preprocess the input image before optimizing
    train_dataset = train_dataset.map(util.parse_py_func_train)
    train_dataset = train_dataset.map(util.parse_dict_train)
    train_dataset = train_dataset.batch(batch_size)

    # Target image read
    test_name = os.path.join(os.getcwd(), '0076.png')
    test_im = cv2.imread(test_name, cv2.IMREAD_COLOR)
    test_im = test_im[:, :, [0]]
    test_im = cv2.normalize(test_im.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    height, width = test_im.shape
    if height != width:
        im_size = np.minimum(height, width)
        test_im = test_im[0 : im_size, 0 : im_size]
    # Prepad the image
    theta = np.arcsin(wavelength / (2 * pp))
    padSize = np.abs(z_prop) * np.tan(theta)
    padN = int(np.ceil(padSize / pp))
    oh, ow = (slm_h - 2 * padN), (slm_w - 2 * padN)
    test_im = cv2.resize(test_im, (oh, ow))
    test_im = np.clip(test_im, 0.0, 1.0)
    test_im = np.pad(test_im, ((padN, padN), (padN, padN)), mode = 'constant')

    # For tensorflow application, we should use 4-dimensional image
    test_im = test_im[np.newaxis, :, :, np.newaxis]

    # Define gradient descent optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    # Log for loss and PSNR
    train_loss_results = []
    train_psnr_results = []

    # Initial phase profile (random phase or double phase encoded phase)
    init_field = Net.ASM2_propagation(-z_prop, pp, wavelength)(tf.dtypes.cast(tf.math.sqrt(test_im), tf.complex64))
    init_dp = Net.encode_double_phase(init_field)

    # Phase variable for optimization
    phase_var = tf.Variable(init_dp.numpy(), name = 'phase_var', dtype = tf.float32)

    # Optimization loop
    for epoch in range(n_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_psnr_avg = tf.keras.metrics.Mean()
        for im_input in train_dataset:
            im_temp = im_input['im_gray']

            # Compute complex wavefront made by phase-only SLM
            hlg = tf.math.exp(1.j * tf.dtypes.cast(phase_var, tf.complex64))

            # Reconstruct
            recon = Net.ASM2_propagation(z_prop, pp, wavelength)(hlg)
            I_r = tf.math.abs(recon) ** 2
            I_rcrop = tf.slice(I_r, [0, padN, padN, 0], [-1, oh, ow, -1])
            I_tcrop = im_temp[:, padN : padN + oh, padN : padN + ow, :]

            # For holographic image comparison, mean value of reconstructed image should be set to be the mean value of target image
            I_rcrop = I_rcrop - tf.math.reduce_mean(I_rcrop, axis = [1, 2], keepdims=True) + np.mean(I_tcrop, axis = (1, 2), keepdims = True)
            I_r = I_r - tf.math.reduce_mean(I_r, axis=[1, 2], keepdims=True) + np.mean(im_temp, axis = (1, 2), keepdims = True)
            I_rcrop = tf.clip_by_value(I_rcrop, 0.0, 1.0)
            I_r = tf.clip_by_value(I_r, 0.0, 1.0)

            # Plot phase profile
            phase_var_np = np.angle(np.exp(1.j * phase_var.numpy()))
            phase_var_np = phase_var_np[0, ...]
            phase_var_np = (phase_var_np + np.pi) / (2 * np.pi)
            phase_var_np = np.uint8(phase_var_np * 255)
            phase_name = "phase_epoch_{:03d}.png".format(epoch)
            phase_name = os.path.join(result_dir, phase_name)
            cv2.imwrite(phase_name, phase_var_np)

            # Plot reconstructed image
            I_r_np = I_r.numpy()
            I_r_np = I_r_np[0, ...]
            I_r_np_crop = I_r_np[padN : padN + oh, padN : padN + ow, :]
            I_r_name = "recon_epoch_{:03d}.png".format(epoch)
            I_r_name = os.path.join(result_dir, I_r_name)
            cv2.imwrite(I_r_name, np.uint8(I_r_np_crop * 255))

            # Compute gradient (dI / dphi) based on Wirtinger derivative
            delta_f = 2 * tf.dtypes.cast((I_r - im_temp), dtype = tf.complex64) * 2 * recon
            delta = Net.ASM2_propagation(-z_prop, pp, wavelength)(delta_f)
            recon_grad = -1.j * tf.math.exp(-1.j * tf.dtypes.cast(phase_var, dtype = tf.complex64)) * delta
            recon_const = tf.math.real(recon_grad)

            # Define loss
            total_grad = [recon_const]
            mse = tf.keras.losses.MeanSquaredError()
            loss_mse = mse(im_temp, I_r)
            loss_psnr = tf.reduce_mean(tf.image.psnr(I_tcrop, I_rcrop.numpy(), max_val = 1))

            # Update via gradient descent
            optimizer.apply_gradients(zip(total_grad, [phase_var]))

            # Track the loss and PSNR
            epoch_loss_avg(loss_mse)
            epoch_psnr_avg(loss_psnr)

        train_loss_results.append(epoch_loss_avg.result())
        train_psnr_results.append(epoch_psnr_avg.result())

        # Print results
        print("Epoch: {:03d} MSE Loss: {:.6f} PSNR: {:.2f}".format(epoch, epoch_loss_avg.result(), epoch_psnr_avg.result()))

if __name__ == "__main__":
    # Check if project dir exists or make a new directory
    util.make_dir(PARAM.get('ROOT'))
    util.make_dir(PARAM.get('RESULT_DIR'))

    # To change tensor to numpy array or string, we should enable eager execution
    tf.compat.v1.enable_eager_execution()
    tf.executing_eagerly()
    main()
