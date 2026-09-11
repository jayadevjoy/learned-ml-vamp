"""
General utilities for complex-valued signal processing.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Union, Tuple


# Unitary FFT along the last axis
@tf.function
def ufft(x: tf.Tensor) -> tf.Tensor:
    nfft = tf.cast(tf.shape(x)[-1], tf.float32)
    fft_scale = tf.cast(tf.sqrt(nfft), x.dtype)
    return tf.signal.fft(x) / fft_scale


# Unitary IFFT along the last axis
@tf.function
def uifft(x: tf.Tensor) -> tf.Tensor:
    nfft = tf.cast(tf.shape(x)[-1], tf.float32)
    fft_scale = tf.cast(tf.sqrt(nfft), x.dtype)
    return tf.signal.ifft(x) * fft_scale


# Add AWGN to a complex signal
def add_awgn(x: tf.Tensor, noise_var: tf.Tensor) -> tf.Tensor:
    return complex_gaussian(tf.shape(x), mean=x, var=tf.cast(noise_var, tf.float32))


# Generates a complex Gaussian tensor
def complex_gaussian(shape: Union[Tuple[int, ...], tf.TensorShape], 
                     mean: Optional[tf.Tensor] = None, 
                     var: float = 1.0) -> tf.Tensor:
    stddev = tf.sqrt(var/2)
    wre = tf.random.normal(shape, mean=0., stddev=stddev)
    wim = tf.random.normal(shape, mean=0., stddev=stddev)
    x = tf.complex(wre, wim)
    if not (mean is None):
        x += mean
    return x


# Compute the Gaussian loss (negative log-likelihood of the Gaussian distribution)
def gaussian_loss(x, x_hat, log_xvar):
    loss = tf.abs(x - x_hat)**2 * tf.exp(-log_xvar) + log_xvar
    loss = tf.reduce_mean(loss) 
    return loss


# Compute the mean squared error between two tensors
def mse(x, x_hat):
    return tf.reduce_mean(tf.abs(x - x_hat)**2)


# Convert a real-valued signal to a complex-valued signal (numerically the same)
def real_to_complex(xre):
    xim = tf.zeros_like(xre)
    return tf.complex(xre, xim)


# Quantizes a complex signal using scalar uniform quantization per component
@tf.function
def quantizer(y_obs: tf.Tensor, delta: tf.Tensor, b: int = 10) -> tf.Tensor:
    qmax = 2**(b - 1) - 1
    qmin = -2**(b - 1)

    re = tf.math.real(y_obs)
    im = tf.math.imag(y_obs)

    q_re = tf.clip_by_value(tf.round(re / delta), qmin, qmax) * delta
    q_im = tf.clip_by_value(tf.round(im / delta), qmin, qmax) * delta
    return tf.complex(q_re, q_im)


# Compute quantizer delta from complex samples using backoff-based method
@tf.function
def delta_backoff(y_obs: tf.Tensor, b: int = 10, backoff_db: float = 12.0) -> tf.Tensor:
    # Estimate power per dimension: E[|y|^2]/2
    pow_per_dim = tf.reduce_mean(tf.abs(y_obs)**2) / 2.0

    # Compute delta from backoff
    A = tf.sqrt(pow_per_dim * tf.pow(10.0, 0.1 * backoff_db))
    delta = A / (2**(b - 1))
    return delta


# Computes Capacity, MSE, and NMSE
def metrics(r_true: tf.Tensor,
            r_hat: tf.Tensor,
            des_known: bool = True,
            des_idx0: Optional[int] = 0,
            des_idx1: Optional[int] = 512) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    
    nfft = tf.shape(r_true)[1]
    x_true = ufft(r_true)
    x_hat = ufft(r_hat)

    if des_known:
        # capacity: subband slice
        x_true_sub = x_true[:, des_idx0:des_idx1]
        x_hat_sub = x_hat[:, des_idx0:des_idx1]

        # mse/nmse: zero-mask + ifft
        mask_1d = tf.concat([tf.zeros(des_idx0, dtype=tf.complex64),
                             tf.ones(des_idx1 - des_idx0, dtype=tf.complex64),
                             tf.zeros(nfft - des_idx1, dtype=tf.complex64)], axis=0)
        mask = tf.broadcast_to(mask_1d, tf.shape(x_true))
        r_true_filt = uifft(x_true * mask)
        r_hat_filt = uifft(x_hat * mask)
    else:
        x_true_sub, x_hat_sub = x_true, x_hat
        r_true_filt, r_hat_filt = r_true, r_hat

    # capacity
    x_true_c = x_true_sub - tf.reduce_mean(x_true_sub, axis=0, keepdims=True)
    x_hat_c = x_hat_sub - tf.reduce_mean(x_hat_sub, axis=0, keepdims=True)

    corr = tf.abs(tf.reduce_mean(x_true_c * tf.math.conj(x_hat_c), axis=0, keepdims=True))
    sigma_1 = tf.sqrt(tf.reduce_mean(tf.abs(x_true_c) ** 2, axis=0, keepdims=True))
    sigma_2 = tf.sqrt(tf.reduce_mean(tf.abs(x_hat_c) ** 2, axis=0, keepdims=True))

    rho_k = corr / (sigma_1 * sigma_2)
    rate_k = -tf.math.log(1 - tf.pow(rho_k, 2)) / tf.math.log(2.0)
    cap = tf.reduce_mean(rate_k)

    # mse / nmse
    mserr = tf.reduce_mean(tf.math.square(tf.abs(r_true_filt - r_hat_filt)))
    mse_per_sample = tf.reduce_sum(tf.abs(r_true_filt - r_hat_filt) ** 2, axis=1)
    power_per_sample = tf.reduce_sum(tf.abs(r_true_filt) ** 2, axis=1)
    nmse = tf.reduce_mean(mse_per_sample / power_per_sample)

    return cap, mserr, nmse