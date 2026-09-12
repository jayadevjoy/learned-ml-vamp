"""
Implementation of the Learned VAMP algorithm for signal estimation with nonlinear saturation effects.
"""

import tensorflow as tf
from typing import Optional
from .utilities import real_to_complex, mse, ufft, uifft
from .nonlinear import SatNeuralEst, SatNL
from .source import SpecNeuralUpdate


class VampSatEst(tf.keras.Model):
    """
    VAMP (Vector Approximate Message Passing) estimator for systems with 
    saturating nonlinearities.

    This class implements a two-denoiser VAMP:
    - Spectral denoiser (`spec_est`): typically fixed, based on a spectral prior.
    - Neural denoiser (`SatNeuralEst`): learned, incorporates saturation effects.

    Parameters
    ----------
    spec_est : tf.keras.Layer
        Fixed spectral estimator used as the input denoiser.
    sat_nl : SatNL
        Module defining the saturating non-linearity.
    niter : int
        Number of VAMP iterations to perform.
    neural_update : bool
        Whether to use a learned message updater (SpecNeuralUpdate) for the VAMP iterations.
    """

    def __init__(self, 
                 spec_est: tf.keras.Layer,
                 sat_nl: SatNL,
                 niter: int = 1,
                 neural_update: bool = True):
        super().__init__(name='VampEst')

        self.niter = niter                    # Number of VAMP iterations
        self.spec_est = spec_est              # Input spectral estimator
        self.sat_nl = sat_nl                  # Saturation non-linearity
        self.neural_update = neural_update    # Whether to use learned message updater

        # Create a learned output denoiser for each VAMP iteration
        self.denoiser_out = [SatNeuralEst(sat_nl) for _ in range(niter)]

        # Create a learned message updater for each VAMP iteration
        if self.neural_update:
            self.msg_updater = [SpecNeuralUpdate(sat_nl) for _ in range(niter)]

    def build(self, input_shapes):
        """
        Build method to initialize the output denoiser and message updater with input shapes.

        Parameters
        ----------
        input_shapes : list of shapes
            Expected to be [psd_shape, rtd_shape], both with matching dimensions.
        """
        if len(input_shapes) != 2 or input_shapes[0][0] != input_shapes[1][0]:
            raise ValueError("Input shapes must match and have length 2.")

        self.nfft = input_shapes[0][0]  # Dimensionality of input

        # Define the input shape for the denoiser
        denoiser_input_shape = ((self.nfft,), (self.nfft,), (self.nfft,))
        updater_input_shape = ((self.nfft,), (self.nfft,), (self.nfft,), (self.nfft,))

        # Build each output denoiser with the specified shape
        for denoiser in self.denoiser_out:
            denoiser.build(denoiser_input_shape)

        # Build each message updater with the specified shape
        if self.neural_update:
            for updater in self.msg_updater:
                updater.build(updater_input_shape)

    def call(self, 
             S: tf.Tensor,
             mu: tf.Tensor,
             y_obs: tf.Tensor,
             x_true: Optional[tf.Tensor] = None,
             train: bool = True,
             des_idx0: Optional[int] = 0, 
             des_idx1: Optional[int] = 512) -> tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Runs the VAMP inference loop.

        Parameters
        ----------
        S : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior variance of the input signal.
        mu : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior mean of the input signal.
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
        loss : tf.Tensor or None
            Computed loss if in training mode, else None.
        """
        # Initial spectral denoising estimate
        z_1, gamma_1 = self.spec_est.est_init(S, mu)

        # Initialize loss accumulator
        loss = 0.0

        if self.neural_update:
            for i, (denoiser, updater) in enumerate(zip(self.denoiser_out, self.msg_updater)):
                # Neural denoising step
                v, gamma_0 = denoiser(z_1, gamma_1, y_obs)
                z_0 = ufft(v)

                # Spectral denoising step
                x_hat, _ = self.spec_est(z_0, gamma_0, S, mu)
                r_hat = uifft(x_hat)

                # Message updates
                beta_0, beta_1, gamma_1 = updater(z_0, gamma_0, S, mu)
                z_1 = beta_0 * r_hat - beta_1 * v

                # Compute loss
                if (i + 1) < self.niter and train:
                    loss += mse(x_true[:, des_idx0:des_idx1], x_hat[:, des_idx0:des_idx1]) * (i + 1) / (self.niter * (self.niter - 1) / 2)
        else:
            for i, denoiser in enumerate(self.denoiser_out):
                # Neural denoising step
                v, gamma_0 = denoiser(z_1, gamma_1, y_obs)
                z_0 = ufft(v)

                # Spectral denoising step
                x_hat, gamma_1 = self.spec_est(z_0, gamma_0, S, mu)
                r_hat = uifft(x_hat)

                # Message updates
                beta_0, beta_1 = 1, 0
                z_1 = beta_0 * r_hat - beta_1 * v

                # Compute loss
                if (i + 1) < self.niter and train:
                    loss += mse(x_true[:, des_idx0:des_idx1], x_hat[:, des_idx0:des_idx1]) * (i + 1) / (self.niter * (self.niter - 1) / 2)

        if train:
            wt = 0.25
            loss = wt * loss + (1 - wt) * mse(x_true[:, des_idx0:des_idx1], x_hat[:, des_idx0:des_idx1])
        else:
            loss = None

        return r_hat, loss


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

    def call(self, 
             psd: tf.Tensor,
             f: tf.Tensor,
             y_obs: tf.Tensor) -> tf.Tensor:
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
        # Construct the IFFT matrix (scaled unitary)
        ifft_mat = uifft(tf.eye(self.nfft, dtype=tf.complex64))

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