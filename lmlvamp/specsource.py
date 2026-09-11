"""
Module for generating a source with a given PSD and a 
linear denoiser based on PSD in the frequency domain.
"""

import numpy as np
import tensorflow as tf
from typing import Optional
from .utilities import real_to_complex, ufft, uifft


class SpecSource(tf.Module):
    """
    Generator for complex-valued signals with specified frequency-domain PSD.
    """

    def __init__(self,
                 nfft: int = 512,
                 src_intervals: Optional[np.ndarray] = None,
                 snr: Optional[np.ndarray] = None,
                 nsamp: int = 100):
        """
        Parameters
        ----------
        nfft : int
            FFT size (number of frequency bins).
        src_intervals : np.ndarray
            Active intervals (rows of [start, end]) in the frequency domain.
        snr : np.ndarray
            Desired SNR in dB for each active interval.
        nsamp : int
            Number of signal samples to generate.
        """
        super().__init__(name="SpecSource")

        self.nfft = nfft
        self.src_intervals = src_intervals if src_intervals is not None else np.array([[0, nfft]])
        self.nsrc = self.src_intervals.shape[0]
        self.nsamp = nsamp

        # Set SNR vector
        self.snr = snr if snr is not None else np.zeros(self.nsrc)
        if len(self.snr) != self.nsrc:
            raise ValueError(f"Length of snr must match number of source intervals: {self.nsrc}")

        # Construct the PSD
        self.psd = np.zeros((self.nfft,))
        for k in range(self.nsrc):
            i0, i1 = self.src_intervals[k]
            if i0 < 0 or i1 > self.nfft or i0 >= i1:
                raise ValueError(f"Invalid src_interval: [{i0}, {i1}) for nfft={self.nfft}")
            psdk = self.nfft * 10 ** (0.1 * self.snr[k]) / (i1 - i0)
            self.psd[i0:i1] = psdk

        self.psd = tf.constant(self.psd, dtype=tf.float32)
        self.psd_mag = tf.sqrt(self.psd / 2)  # Convert to complex std deviation

    def __call__(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate random signal samples with the desired PSD.

        Returns
        -------
        x : tf.Tensor (nsamp, nfft), complex
            Frequency-domain samples.
        r : tf.Tensor (nsamp, nfft), complex
            Corresponding time-domain samples (via IFFT).
        """
        x_re = self.psd_mag * tf.random.normal((self.nsamp, self.nfft), dtype=tf.float32)
        x_im = self.psd_mag * tf.random.normal((self.nsamp, self.nfft), dtype=tf.float32)
        x = tf.complex(x_re, x_im)
        r = uifft(x)
        return x, r

    def measure_snr(self, r: tf.Tensor, r_hat: tf.Tensor) -> np.ndarray:
        """
        Measure SNR per interval between true and estimated time-domain signals.

        Parameters
        ----------
        r : tf.Tensor
            Ground truth signal.
        r_hat : tf.Tensor
            Estimated signal.

        Returns
        -------
        snr_est : np.ndarray
            Estimated SNR (in dB) for each source interval.
        """
        x = ufft(r)
        x_hat = ufft(r_hat)

        x, x_hat, psd = x.numpy(), x_hat.numpy(), self.psd.numpy()
        snr_est = np.zeros(self.nsrc)
        for k in range(self.nsrc):
            i0, i1 = self.src_intervals[k]
            errk = np.abs(x[:, i0:i1] - x_hat[:, i0:i1]) ** 2
            snr_est[k] = 10 * np.log10(np.mean(psd[i0:i1]) / np.mean(errk))
        return snr_est


class SpecEstim(tf.keras.layers.Layer):
    """
    Linear spectral denoiser using known or estimated PSD.
    """

    def __init__(self):
        super().__init__(name='SpecEstim')

    def call(self,
             z_0: tf.Tensor,
             gamma_0: tf.Tensor,
             S: tf.Tensor,
             mu: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Apply frequency-domain Wiener filtering using PSD.

        Parameters
        ----------
        z_0 : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Posterior mean of the input frequency-domain signal.
        gamma_0 : tf.Tensor, shape (nsamp, 1), tf.float32
            Posterior precision (inverse variance).
        S : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior variance of the input signal.
        mu : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior mean of the input signal.

        Returns
        -------
        x_hat : tf.Tensor
            Estimated frequency-domain signal.
        gamma_1 : tf.Tensor
            Posterior precision (inverse variance) after denoising.
        """
        gain = S * gamma_0 / (S * gamma_0 + 1)
        gain_c = real_to_complex(gain)

        x_hat = mu + gain_c * (z_0 - mu)
        x_var_post = gain / gamma_0
        x_var_post = tf.reduce_mean(x_var_post, axis=1, keepdims=True)
        gamma_1 = 1.0 / x_var_post

        return x_hat, gamma_1

    def est_init(self,
                 S: tf.Tensor,
                 mu: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize estimates to zero signal with PSD as variance.

        Parameters
        ----------
        S : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior variance of the input signal.
        mu : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior mean of the input signal.

        Returns
        -------
        z_1 : tf.Tensor
            Initialized time-domain estimate.
        gamma_1 : tf.Tensor
            Posterior precision (inverse variance per sample, scalar).
        """

        z_1 = uifft(mu)
        x_var_post = tf.reduce_mean(S, axis=1, keepdims=True)
        gamma_1 = 1.0 / x_var_post

        return z_1, gamma_1