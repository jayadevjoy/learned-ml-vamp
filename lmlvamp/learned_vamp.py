"""
Implementation of the Learned VAMP algorithm for signal estimation with nonlinear saturation effects.
"""

import numpy as np
import tensorflow as tf
from typing import Optional
from .utilities import real_to_complex, mse, ufft, uifft
from .nonlin import SatNeuralEst, SatNL


class VampSatEst(tf.keras.Model):
    """
    VAMP (Vector Approximate Message Passing) estimator for systems with 
    saturating nonlinearities.

    This class implements a two-denoiser VAMP:
    - Input denoiser (`spec_est`): typically fixed, based on a spectral prior.
    - Output denoiser (`SatNeuralEst`): learned, incorporates saturation effects.

    Parameters
    ----------
    spec_est : tf.keras.Layer
        Fixed spectral estimator used as the input denoiser.
    sat_nl : SatNL
        Module defining the saturating non-linearity.
    niter : int
        Number of VAMP iterations to perform.
    """

    def __init__(self, 
                 spec_est: tf.keras.Layer,
                 sat_nl: SatNL,
                 niter: int = 1):
        super().__init__(name='VampEst')

        self.niter = niter                    # Number of VAMP iterations
        self.spec_est = spec_est              # Input spectral estimator
        self.sat_nl = sat_nl                  # Saturation non-linearity

        # Create a learned output denoiser for each VAMP iteration
        self.denoiser_out = [SatNeuralEst(sat_nl) for _ in range(niter)]

    def build(self, input_shapes):
        """
        Build method to initialize the output denoisers with input shapes.

        Parameters
        ----------
        input_shapes : list of shapes
            Expected to be [psd_shape, rtd_shape], both with matching dimensions.
        """
        if len(input_shapes) != 2 or input_shapes[0][0] != input_shapes[1][0]:
            raise ValueError("Input shapes must match and have length 2.")

        self.nfft = input_shapes[0][0]  # Dimensionality of input

        # Define the input shape for the denoiser
        denoiser_input_shape = ((self.nfft,), (self.nfft,), (self.nfft,))  # Shape for (x, v, rtd)

        # Build each output denoiser with the specified shape
        for denoiser in self.denoiser_out:
            denoiser.build(denoiser_input_shape)  

    def call(self, 
             psd: tf.Tensor,
             y_obs: tf.Tensor,
             x_true: Optional[tf.Tensor] = None,
             train: bool = True,
             des_idx0: Optional[int] = 0, 
             des_idx1: Optional[int] = 512) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Runs the VAMP inference loop.

        Parameters
        ----------
        psd : tf.Tensor
            Power spectral density for each sample.
        y_obs : tf.Tensor
            Received time-domain signal after non-linearity.
        x_true : tf.Tensor, optional
            True Signal in frequency domain (Interferer used for training - if known, Desried signal used for loss)
        train : bool, default=True
            Whether to run in training mode (affects loss computation)
        des_idx0: int (default=0)
            Start index of the desired frequency subband
        des_idx1: int (default=512)
            End index of the desired frequency subband

        Returns
        -------
        r_hat : tf.Tensor
            Final signal estimate after niter iterations.
        x_var : tf.Tensor
            Final posterior variance estimate.
        """
        # Initial spectral denoising estimate
        r_hat0, x_var0 = self.spec_est.est_init(psd, x_true)

        # Initialize loss accumulator
        loss = 0.0

        for i, denoiser in enumerate(self.denoiser_out):

            # Nonlinear denoising step
            r_hat1, x_var1 = denoiser(r_hat0, x_var0, y_obs)

            # Spectral denoising step
            r_hat0, x_var0 = self.spec_est(r_hat1, x_var1, psd, x_true)

            # Compute update factor 'a' and clip its values to avoid instability
            a = x_var0 / (x_var1 + 1e-3)
            a = tf.clip_by_value(a, 0.01, 0.99)
            ac = real_to_complex(a)  # Convert 'a' to complex if needed

            # Update variance and estimate for next iteration
            r_hat0 = (r_hat1 - ac * r_hat0) / (1 - ac)
            x_var0 = a * x_var0 / (1 - a)

            # Compute loss
            if (i + 1) < self.niter and train:
                x_hat = ufft(r_hat1)
                loss += mse(x_true[:, des_idx0:des_idx1], x_hat[:, des_idx0:des_idx1]) * (i + 1) / (self.niter * (self.niter - 1) / 2)

        if train:
            wt = 0.25
            x_hat = ufft(r_hat1)
            loss = wt * loss + (1 - wt) * mse(x_true[:, des_idx0:des_idx1], x_hat[:, des_idx0:des_idx1])

        return (r_hat1, x_var1, loss) if train else (r_hat1, x_var1)


class OracleLinEst(tf.keras.Model):
    """
    Linear estimator with oracle gain for systems affected by soft saturation nonlinearity.

    Assumes the following receiver model:
        y = f * (r + w_a) + w_b,
        where r = IFFT(x), and x ~ CN(0, psd)

    This module estimates the input signal assuming perfect knowledge of the nonlinear gain f.

    Parameters
    ----------
    sat_nl : SatNL
        Instance of the saturation nonlinearity model (provides noise levels).
    nfft : int, default=512
        FFT size for transforming between time and frequency domains.
    """

    def __init__(self, sat_nl: 'SatNL', nfft: int = 512):
        super().__init__(name='SatLinEst')

        # Store model parameters
        self.sat_nl = sat_nl
        self.var_wa = sat_nl.var_wa
        self.var_wb = sat_nl.var_wb
        self.nfft = nfft
        self.fft_scale = tf.constant(np.sqrt(nfft), dtype=tf.complex64)

    def call(self, 
             psd: tf.Tensor,
             f: tf.Tensor,
             y_obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the LMMSE estimate of the input signal under an oracle linearization of the saturated system.

        Assumed system model:
            y = A x + w
        where:
            A = diag(f) · IDFT
            cov(w) = diag(q) = diag(f² * var_wa + var_wb)

        The LMMSE solution is:
            x̂ = Aᴴ diag(psd) [A diag(psd) Aᴴ + diag(q)]⁻¹ y
            Var(x̂) = diag(psd) - Aᴴ diag(psd) [A diag(psd) Aᴴ + diag(q)]⁻¹ A diag(psd)

        Parameters
        ----------
        psd : tf.Tensor, shape (nsamp, nfft), dtype=tf.float32  
            Power spectral density of the input frequency-domain signal x.
        f : tf.Tensor, shape (nsamp, nfft), dtype=tf.float32  
            Saturation gain f
        y_obs : tf.Tensor, shape (nsamp, nfft), dtype=tf.complex64  
            Received time-domain signal after nonlinearity and noise

        Returns
        -------
        r_hat : tf.Tensor, shape (nsamp, nfft), dtype=tf.complex64
            Estimated time-domain signal corresponding to IFFT(x̂)
        """

        nsamp, nfft = psd.shape

        # Construct the IFFT matrix (scaled unitary)
        ifft_mat = uifft(tf.eye(nfft, dtype=tf.complex64))

        # Compute noise covariance: q = ftd^2 * wvar0 + wvar1
        q = f**2 * self.var_wa + self.var_wb

        # Convert to complex type for broadcasting/multiplication
        fc = real_to_complex(f)         # (nsamp, nfft)
        psdc = real_to_complex(psd)       # (nsamp, nfft)
        qc = real_to_complex(q)           # (nsamp, nfft)

        # Construct A = diag(ftd) * ifft_mat
        A = fc[:, :, None] * ifft_mat[None, :, :]  # (nsamp, nfft, nfft)

        # Precompute A * diag(psd)
        AP = A * psdc[:, None, :]  # (nsamp, nfft, nfft)

        # Compute the covariance matrix Q = A P Aᵀ + diag(q)
        Q = tf.matmul(AP, A, adjoint_b=True) + tf.linalg.diag(qc)  # (nsamp, nfft, nfft)

        # Solve Q z = y to get intermediate z
        z = tf.linalg.solve(Q, y_obs[:, :, None])  # (nsamp, nfft, 1)

        # Estimate x̂ = Aᵀ P z
        x_hat = tf.matmul(AP, z, adjoint_a=True)  # (nsamp, nfft, 1)
        x_hat = tf.squeeze(x_hat, axis=-1)      # (nsamp, nfft)

        # Convert back to time-domain: r_hat = IFFT(x̂)
        r_hat = uifft(x_hat)

        return r_hat


# class LearnedVampEst(tf.keras.Model):
#     """
#     Learned VAMP variant with a trainable neural network as output denoiser.

#     The input denoiser is fixed, and the output denoiser learns
#     nonlinear MMSE behavior from data.

#     Parameters
#     ----------
#     spec_est : tf.keras.Layer
#         Input denoiser (non-trainable)
#     sat_nl : SatNL
#         Module defining the saturating non-linearity.
#     nhid : int
#         Hidden layer size for the learned NN
#     niter : int
#         Number of VAMP iterations (currently only 1 is used)
#     """

#     def __init__(self, 
#                  spec_est: tf.keras.layers.Layer, 
#                  sat_nl: 'SatNL',
#                  nhid=64, 
#                  niter=1):
#         super().__init__(name='LearnedVampEst')

#         self.spec_est = spec_est
#         self.sat_nl = sat_nl
#         self.niter = niter

#         # Define output denoising neural network
#         self.dense1 = tf.keras.layers.Dense(nhid, activation='sigmoid')
#         self.dense2 = tf.keras.layers.Dense(3, activation=None)

#     def call(self, 
#              psd: tf.Tensor,
#              rtd: tf.Tensor,
#              xfd: Optional[tf.Tensor] = None) -> tuple[tf.Tensor, tf.Tensor]:
#         """
#         Performs one iteration of learned VAMP.

#         Parameters
#         ----------
#         psd : tf.Tensor
#             Input denoiser parameter
#         rtd : tf.Tensor
#             Output denoiser parameter (post non-linearity)
#         xfd : tf.Tensor, optional
#             Interferer if known

#         Returns
#         -------
#         xest0 : tf.Tensor
#             Final estimated signal
#         xvar0 : tf.Tensor
#             Final estimated variance
#         """
#         # Initial estimate using input denoiser
#         if self.spec_est.intf_known:
#             xest0, xvar0 = self.spec_est.est_init(psd, xfd)
#         else:
#             xest0, xvar0 = self.spec_est.est_init(psd)

#         nsamp, ntd = xest0.shape
#         Esat = self.sat_nl.Esat

#         for _ in range(self.niter):
#             # Flatten data for NN input
#             xest0_flat = tf.reshape(xest0, (-1, 1))
#             xvar0_flat = tf.reshape(tf.tile(xvar0, (1, ntd)), (-1, 1))
#             rtd_flat = tf.reshape(rtd, (-1, 1))

#             # Concatenate features: |xest0|, xvar0, |rtd|
#             features = tf.concat([
#                 tf.abs(xest0_flat)/np.sqrt(Esat),
#                 xvar0_flat/Esat,
#                 tf.abs(rtd_flat)/np.sqrt(Esat)
#             ], axis=1)

#             # Pass through neural network
#             hidden = self.dense1(features)
#             output = self.dense2(hidden)

#             # Extract NN outputs
#             gain0 = real_to_complex(tf.expand_dims(output[:, 0], axis=1))
#             gain1 = real_to_complex(tf.expand_dims(output[:, 1], axis=1))
#             log_xvar = tf.expand_dims(output[:, 2], axis=1)

#             # Apply VAMP update rule
#             xest0_flat = xest0_flat + gain0 * (rtd_flat - gain1 * xest0_flat)
#             xvar0_flat = tf.exp(log_xvar) * xvar0_flat

#             # Reshape estimates
#             xest0 = tf.reshape(xest0_flat, (nsamp, ntd))
#             xvar0 = tf.reduce_mean(tf.reshape(xvar0_flat, (nsamp, ntd)), axis=1, keepdims=True)

#             # Input denoiser update
#             if self.spec_est.intf_known:
#                 xest, xvar = self.spec_est(xest0, xvar0, psd, xfd)
#             else:
#                 xest, xvar = self.spec_est(xest0, xvar0, psd)
            
#             # Compute update factor 'a' and clip its values to avoid instability
#             a = xvar / (xvar0 + 1e-3)
#             a = tf.clip_by_value(a, 0.01, 0.99)
#             ac = real_to_complex(a)  # Convert 'a' to complex if needed

#             # Update variance and estimate for next iteration
#             xvar0 = a * xvar / (1 - a)
#             xest0 = (xest0 - ac * xest) / (1 - ac)

#         return xest, xvar