"""
General utilities for complex-valued signal processing.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Union, Tuple

def complex_gaussian(
        shape: Union[Tuple[int, ...], tf.TensorShape], 
        mean: Optional[tf.Tensor] = None, 
        var: float = 1.0) -> tf.Tensor:
    """
    Generates a complex Gaussian tensor.
    
    Parameters:
    -----------
    shape:  tuple or tf.TensorShape
        Shape of the output tensor.    
    mean: tf.Tensor of dtype=tf.complex64 or None 
        Mean of the Gaussian.  If None, the mean is 0.
    var: float
        Variance of the Gaussian distribution.
    """
    stddev = tf.sqrt(var/2)
    wre = tf.random.normal(shape, mean=0., stddev=stddev)
    wim = tf.random.normal(shape, mean=0., stddev=stddev)
    x = tf.complex(wre, wim)
    if not (mean is None):
        x += mean
    return x

def gaussian_loss(x, x_hat, log_xvar):
    """
    Compute the Gaussian loss.
    This is the negative log-likelihood of the Gaussian distribution.
    """
    loss = tf.abs(x - x_hat)**2 * tf.exp(-log_xvar) + log_xvar
    loss = tf.reduce_mean(loss) 
    return loss

def mse(x, x_hat):
    """
    Compute the mean squared error between two tensors.
    """
    loss = tf.reduce_mean(tf.abs(x - x_hat)**2)
    return loss

def real_to_complex(xre):
    """
    Convert a real-valued signal to a complex-valued signal.
    The real part is the real part of the signal, and the imaginary
    part is the imaginary part of the signal.
    """
    xim = tf.zeros_like(xre)
    x = tf.complex(xre, xim)
    return x

def corr_cap(r_true: tf.Tensor, 
             r_hat: tf.Tensor, 
             des_known: bool = True,
             des_idx0: Optional[int] = 0, 
             des_idx1: Optional[int] = 512) -> tf.Tensor:
    """
    Computes a lower bound on channel capacity using the correlation method:
        C >= -log2(1 - ρ)

    Parameters:
    - r_true: tf.Tensor of shape (nsamp, nfft), complex-valued ground-truth signal
    - r_hat: tf.Tensor of shape (nsamp, nfft), complex-valued estimated signal
    - des_known: Boolean, whether the desired signal bandwidth is known
    - des_idx0: Starting index of the desired frequency band
    - des_idx1: Ending index of the desired frequency band

    Returns:
    - tf.Tensor: Scalar lower bound on channel capacity in bits
    """

    # Determine FFT size from input
    nfft = r_true.shape[1]

    # Create a normalization constant to make FFT unitary
    fft_scale = tf.constant(np.sqrt(nfft), dtype=tf.complex64)

    # Transform time-domain signals to frequency domain using FFT
    x_true = tf.signal.fft(r_true) / fft_scale
    x_hat = tf.signal.fft(r_hat) / fft_scale

    # If desired signal's frequency range is known, isolate that subband
    if des_known:
        x_true = x_true[:, des_idx0:des_idx1]
        x_hat = x_hat[:, des_idx0:des_idx1]
    
    # Remove mean across samples to ensure zero-mean signals
    x_true -= tf.reduce_mean(x_true, axis=0, keepdims=True)
    x_hat -= tf.reduce_mean(x_hat, axis=0, keepdims=True)

    # Compute correlation and standard deviation
    corr = tf.abs(tf.reduce_mean(x_true * tf.math.conj(x_hat)))
    sigma_1 = tf.sqrt(tf.reduce_mean(tf.abs(x_true) ** 2))
    sigma_2 = tf.sqrt(tf.reduce_mean(tf.abs(x_hat) ** 2))

    # Compute normalized correlation (ρ)
    rho = corr / (sigma_1 * sigma_2)

    # Calculate capacity lower bound: -log2(1 - ρ)
    cap = -tf.math.log(1 - (rho**2)) / tf.math.log(2.0)

    return cap

def error_metric(r_true: tf.Tensor, 
        r_hat: tf.Tensor, 
        des_known: bool = True,
        des_idx0: Optional[int] = 0, 
        des_idx1: Optional[int] = 512) -> tf.Tensor:
    """
    Computes mean squared error (MSE) and normalized MSE (NMSE) between 
    true and estimated complex-valued signals.

    Parameters:
    - r_true: tf.Tensor of shape (nsamp, nfft)
        Ground-truth complex-valued signal in time domain
    - r_hat: tf.Tensor of shape (nsamp, nfft)
        Estimated complex-valued signal in time domain
    - des_known: bool (default=True)
        Indicates if the desired frequency band is known
    - des_idx0: int (default=0)
        Start index of the desired frequency subband
    - des_idx1: int (default=512)
        End index of the desired frequency subband

    Returns:
    - mse: tf.Tensor
        Mean squared error between true and estimated signals
    - nmse: tf.Tensor
        Normalized mean squared error relative to the signal power
    """

    # Get the FFT size from the input tensor's second dimension
    nfft = r_true.shape[1]

    # Compute the FFT normalization constant for unitary FFT scaling
    fft_scale = tf.constant(np.sqrt(nfft), dtype=tf.complex64)

    if des_known:
        # Convert time-domain signals to frequency-domain using FFT
        x_true = tf.signal.fft(r_true) / fft_scale
        x_hat = tf.signal.fft(r_hat) / fft_scale

        # Create a binary mask to isolate the desired frequency subband
        mask_1d = tf.concat([
            tf.zeros(des_idx0, dtype=tf.complex64),
            tf.ones(des_idx1 - des_idx0, dtype=tf.complex64),
            tf.zeros(nfft - des_idx1, dtype=tf.complex64)
        ], axis=0)

        # Broadcast the mask to match the shape of the FFT output
        mask = tf.broadcast_to(mask_1d, tf.shape(x_true))

        # Apply the mask to both true and estimated frequency-domain signals
        x_true = x_true * mask
        x_hat = x_hat * mask

        # Convert the filtered frequency-domain signals back to time-domain
        r_true = fft_scale * tf.signal.ifft(x_true)
        r_hat = fft_scale * tf.signal.ifft(x_hat)

    # Compute Mean Squared Error (MSE)
    mse = tf.reduce_mean(tf.math.square(tf.abs(r_true - r_hat)))

    # Compute Normalized Mean Squared Error (NMSE)
    mse_per_sample = tf.reduce_sum(tf.abs(r_true - r_hat)**2, axis=1)
    power_per_sample = tf.reduce_sum(tf.abs(r_true)**2, axis=1)
    nmse = tf.reduce_mean(mse_per_sample / power_per_sample)

    return mse, nmse

def quantizer(y_obs: tf.Tensor, delta: float, b: int = 10) -> tf.Tensor:
    """
    Quantizes a complex signal using scalar uniform quantization per component.

    Args:
        y_obs : tf.Tensor, tf.complex64
        delta : float scalar, step size
        b : int, number of bits per dimension

    Returns:
        tf.Tensor, tf.complex64, quantized tensor
    """
    max = 2**(b - 1) - 1
    min = -2**(b - 1)

    re = tf.math.real(y_obs)
    im = tf.math.imag(y_obs)

    q_re = tf.clip_by_value(tf.round(re / delta), min, max) * delta
    q_im = tf.clip_by_value(tf.round(im / delta), min, max) * delta

    return tf.complex(q_re, q_im)

def delta_backoff(y_obs: tf.Tensor, b: int = 10, backoff_db: float = 12.0) -> float:
    """
    Compute quantizer delta from complex samples using backoff-based method.

    Args:
        y_obs : tf.Tensor, shape (nsamps, nfft), dtype=tf.complex64
            Complex-valued samples to be quantized.
        b : int
            Number of bits per dimension (real and imaginary).
        backoff_db : float
            Backoff in dB to scale quantizer range above average signal power.

    Returns:
        delta : float (NumPy float32)
            Step size for quantization.
    """
    # Estimate power per dimension: E[|y|^2]/2
    pow = tf.reduce_mean(tf.abs(y_obs)**2)
    pow = pow / 2.0

    # Compute delta from backoff
    A = tf.sqrt(pow * tf.pow(10.0, 0.1 * backoff_db))
    delta = A / (2**(b - 1))

    return delta.numpy()