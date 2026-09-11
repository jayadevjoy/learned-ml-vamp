"""
Module for generating a source with a given PSD and a linear denoiser based on PSD in the frequency domain.
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional
from complex_utils import real_to_complex


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

        # FFT scaling to make IFFT unitary
        self.fft_scale = tf.constant(np.sqrt(self.nfft), dtype=tf.complex64)

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
        r = self.fft_scale * tf.signal.ifft(x)
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
        x = tf.signal.fft(r) / self.fft_scale
        x_hat = tf.signal.fft(r_hat) / self.fft_scale

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

    def __init__(self,
                 nfft: int = 512,
                 intf_known: bool = True,
                 intf_idx0: Optional[int] = None,
                 intf_idx1: Optional[int] = None):
        """
        Parameters
        ----------
        nfft : int
            FFT size.
        intf_known : bool
            If True, replace estimated values in known interferer region with ground truth.
        intf_idx0 : int
            Start index of interferer region.
        intf_idx1 : int
            End index of interferer region.
        """
        super().__init__(name='SpecEstim')
        self.nfft = nfft
        self.intf_known = intf_known
        self.intf_idx0 = intf_idx0
        self.intf_idx1 = intf_idx1
        self.fft_scale = tf.constant(np.sqrt(self.nfft), dtype=tf.complex64)

    def call(self,
             r_mean: tf.Tensor,
             x_var: tf.Tensor,
             psd: tf.Tensor,
             x_true: Optional[tf.Tensor] = None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Apply frequency-domain Wiener filtering using PSD.

        Parameters
        ----------
        r_mean : tf.Tensor
            Prior mean of time-domain signal.
        x_var : tf.Tensor
            Prior variance of signal (shared across all bins).
        psd : tf.Tensor
            Known or estimated power spectral density.
        x_true : tf.Tensor (optional)
            Ground-truth frequency-domain signal (used if interferer region is known).

        Returns
        -------
        r_hat : tf.Tensor
            Estimated time-domain signal (via IFFT).
        x_var_post : tf.Tensor
            Posterior variance per sample (averaged across frequency bins).
        """
        x_mean = tf.signal.fft(r_mean) / self.fft_scale
        gain = psd / (psd + x_var)
        gain_c = real_to_complex(gain)
        x_hat = gain_c * x_mean
        x_var_post = gain * x_var

        if self.intf_known:
            mask_1d = tf.concat([
                tf.zeros(self.intf_idx0, dtype=tf.float32),
                tf.ones(self.intf_idx1 - self.intf_idx0, dtype=tf.float32),
                tf.zeros(self.nfft - self.intf_idx1, dtype=tf.float32)
            ], axis=0)
            mask = tf.broadcast_to(mask_1d, tf.shape(x_hat))
            mask_c = tf.cast(mask, tf.complex64)

            x_hat = x_hat * (1 - mask_c) + x_true * mask_c
            x_var_post = x_var_post * (1 - mask)

        r_hat = self.fft_scale * tf.signal.ifft(x_hat)
        x_var_post = tf.reduce_mean(x_var_post, axis=1, keepdims=True)

        self.x_hat = x_hat
        self.x_var_post = x_var_post

        return r_hat, x_var_post

    def est_init(self,
                 psd: tf.Tensor,
                 x_true: Optional[tf.Tensor] = None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize estimates to zero signal with PSD as variance.

        Parameters
        ----------
        psd : tf.Tensor
            Power spectral density for each sample.
        x_true : tf.Tensor (optional)
            Ground-truth frequency-domain signal (used if interferer region is known).

        Returns
        -------
        r_hat : tf.Tensor
            Zero-initialized time-domain estimate.
        x_var_post : tf.Tensor
            Posterior variance estimate per sample (scalar).
        """
        x_hat = tf.zeros(psd.shape, dtype=tf.complex64)
        x_var_post = psd

        if self.intf_known:
            mask_1d = tf.concat([
                tf.zeros(self.intf_idx0, dtype=tf.float32),
                tf.ones(self.intf_idx1 - self.intf_idx0, dtype=tf.float32),
                tf.zeros(self.nfft - self.intf_idx1, dtype=tf.float32)
            ], axis=0)
            mask = tf.broadcast_to(mask_1d, tf.shape(x_hat))
            mask_c = tf.cast(mask, dtype=tf.complex64)

            x_hat = x_hat * (1 - mask_c) + x_true * mask_c
            x_var_post = x_var_post * (1 - mask)

        r_hat = self.fft_scale * tf.signal.ifft(x_hat)
        x_var_post = tf.reduce_mean(x_var_post, axis=1, keepdims=True)

        self.x_hat = x_hat
        return r_hat, x_var_post