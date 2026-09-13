"""
Simulation of the VAMP algorithm under saturation non-linearity
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Import custom modules and components
from .nonlinear import SatNL
from .source import SpecSource, SpecEstim
from .utilities import quantizer, delta_backoff, ufft, uifft
from .lmlvamp import VampSatEst, OracleLinEst


class VampSim:
    """
    Simulation and training of a Learned VAMP estimator for nonlinear spectral signal recovery
    using a saturation nonlinearity model.
    """

    def __init__(self, 
                 nfft=512, 
                 nsamp=100, 
                 nitvamp=1,  
                 snr=None, 
                 inr=None, 
                 snr_interval=None, 
                 inr_interval=None,
                 intf_known=True,
                 quantize=False):
        """
        Initialize simulation environment and models.

        Parameters
        ----------
        nfft : int
            FFT size (signal dimensionality).
        nsamp : int
            Number of samples per training batch.
        nitvamp : int
            Number of VAMP iterations.
        snr : float
            SNR (in dB) for desired signal.
        inr : float
            INR (in dB) for interferer signal.
        snr_interval : np.ndarray
            Active interval ([start, end]) in the frequency domain of desired signal.
        inr_interval : np.ndarray
            Active interval ([start, end]) in the frequency domain of interferer signal.
        intf_known : bool
            Flag indicating if interferer band is known at the receiver.
        quantize : bool
            Whether to quantize the signal.
        """
        self.nfft = nfft
        self.nsamp = nsamp
        self.intf_known = intf_known
        self.nitvamp = nitvamp
        self.quantize = quantize

        if snr_interval is None:
            snr_interval = np.array([0, 100])

        if inr_interval is None:
            inr_interval = np.array([300, 400])

        if snr is None:
            snr = 10

        if inr is None:
            inr = 30
        
        self.snr_inr = np.array([snr, inr])
        self.src_intervals = np.array([snr_interval, inr_interval])

        # Interferer interval 
        self.intf_idx0, self.intf_idx1 = inr_interval

        # Desired signal interval
        self.des_idx0, self.des_idx1 = snr_interval

        # Identifying the interferer band in the frequency domain
        mask_1d = tf.concat([tf.zeros(self.intf_idx0, dtype=tf.float32),
                             tf.ones(self.intf_idx1 - self.intf_idx0, dtype=tf.float32),
                             tf.zeros(self.nfft - self.intf_idx1, dtype=tf.float32)], axis=0)
        self.mask = tf.broadcast_to(mask_1d, (self.nsamp, self.nfft))
        self.mask_c = tf.cast(self.mask, dtype=tf.complex64)

        # Create spectral source model
        self.src = SpecSource(nfft=self.nfft, src_intervals=self.src_intervals, snr=self.snr_inr, nsamp=self.nsamp)

        # Create nonlinear channel model with saturation
        self.sat_nl = SatNL(noise0_db=0, noise1_db=-10, sat_db=40)

        # Create spectral estimator (input denoiser for VAMP)
        self.spec_est = SpecEstim(nfft=self.nfft)

        # Instantiate VAMP estimator using the spectral denoiser
        self.vamp_est = VampSatEst(niter=self.nitvamp, spec_est=self.spec_est, sat_nl=self.sat_nl)

    def train(self, nsteps=500):
        """
        Train the Learned VAMP estimator using generated data.

        Parameters
        ----------
        nsteps : int
            Number of training epochs.
        """
        # Input to VAMP estimator is (psd, rtd), each of shape (nsamp, nfft)
        input_shapes = ((self.nfft,), (self.nfft,))
        self.vamp_est.build(input_shapes)

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.9,
            staircase=True)

        # Setup optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_hist = []

        # Initialize best loss and weights
        # This is used to save the best model during training
        best_loss = float('inf')
        best_weights = None

        for epoch in range(nsteps):
            # Generate clean signal and its time-domain version
            x, r = self.src()

            # Pass through nonlinear channel with saturation
            y, f = self.sat_nl(r)

            # Generate input PSD replicated for each sample
            psd = tf.tile(self.src.psd[None, :], (self.nsamp, 1))

            if self.intf_known:
                S = psd * (1 - self.mask)
                mu = x * self.mask_c
            else:
                S = psd
                mu = tf.zeros(S.shape, dtype=tf.complex64)

            # If quantization is enabled, apply quantization to the received time-domain signal
            if self.quantize:
                delta = delta_backoff(y)
                y = quantizer(y, delta)

            with tf.GradientTape() as tape:
                # Forward pass through VAMP
                r_hat, loss = self.vamp_est(S=S, mu=mu, y_obs=y, x_true=x, train=True, des_idx0=self.des_idx0, des_idx1=self.des_idx1)

            # Loss
            self.loss_hist.append(loss.numpy())

            # Backpropagation and optimizer step
            variables = self.vamp_est.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            # Save best weights in memory
            if loss.numpy() < best_loss:
                best_loss = loss.numpy()
                best_weights = self.vamp_est.get_weights()

            if epoch % 100 == 0:
                print(f"[Epoch: {epoch}] Loss: {loss.numpy():.4f}; Best Loss: {best_loss:.4f}")

        # # Plot training loss curve
        # plt.plot(self.loss_hist)
        # plt.title("Training Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.grid(True)
        # plt.show()

        # Restore best weights
        self.vamp_est.set_weights(best_weights)
        print("Training complete. Best weights restored.")

        # # Save best weights
        # filepath = os.path.join(os.getcwd(), 'vamp_est.weights.h5')
        # self.vamp_est.save_weights(filepath)

    def evaluate(self, load_model=True):
        """
        Evaluate the trained VAMP estimator on a new test batch.

        Parameters
        ----------
        load_model : bool
            Whether to load the trained model weights from file.

        Returns
        -------
        r : tf.Tensor
            Ground truth time-domain signal.
        r_hat_vamp : tf.Tensor
            Estimated signal by VAMP.
        r_hat_lin : tf.Tensor
            Linear spectral estimator output.
        r_hat_orc : tf.Tensor
            Oracle linear estimate output.
        """
        if load_model:
            self.vamp_est = VampSatEst(niter=self.nitvamp, spec_est=self.spec_est, sat_nl=self.sat_nl)
            if isinstance(self.vamp_est, VampSatEst):
                input_shapes = ((self.nfft,), (self.nfft,))
                self.vamp_est.build(input_shapes)
            filepath = os.path.join(os.getcwd(), 'vamp_est.weights.h5')
            if os.path.exists(filepath):
                print("Loading the model...")
                self.vamp_est.load_weights(filepath)
            else:
                raise FileNotFoundError(f"{filepath} not found. Run sim.train() to train model.")

        # Generate new data
        x, r = self.src()
        
        # Pass through nonlinear channel with saturation
        y, f = self.sat_nl(r)

        # Generate input PSD replicated for each sample
        psd = tf.tile(self.src.psd[None, :], (self.nsamp, 1))

        if self.intf_known:
            S = psd * (1 - self.mask)
            mu = x * self.mask_c
        else:
            S = psd
            mu = tf.zeros(S.shape, dtype=tf.complex64)

        # If quantization is enabled, apply quantization to the received time-domain signal
        if self.quantize:
            delta = delta_backoff(y)
            y = quantizer(y, delta)

        # Run VAMP
        r_hat_vamp = self.vamp_est(S=S, mu=mu, y_obs=y, x_true=x, train=False)

        # Get linear baseline estimate
        wvar = self.sat_nl.var_wa + self.sat_nl.var_wb
        y_var = tf.ones((self.nsamp, 1)) * wvar
        y_freq = ufft(y)
        x_hat_lin, _ = self.spec_est(y_freq, (1/y_var), S, mu)
        r_hat_lin = uifft(x_hat_lin)

        # Get oracle linear estimate
        self.oracle = OracleLinEst(sat_nl=self.sat_nl, nfft=self.nfft)  
        r_hat_orc = self.oracle(psd, f, y)

        return r, r_hat_vamp, r_hat_lin, r_hat_orc