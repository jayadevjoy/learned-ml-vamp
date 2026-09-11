"""
Saturation nonlinearity and its linear and neural network estimators
"""

import numpy as np
import tensorflow as tf
from .utilities import real_to_complex, add_awgn


class SatNL(tf.Module): 
    """
    Receiver front-end with saturation nonlinearity:
        u[i] = r[i] + w_a[i]
        ysat[i] = f(|u[i]|/√Psat) * u[i]
        y[i] = ysat[i] + w_b[i]
    """
    def __init__(self, noise0_db=0, noise1_db=-10, sat_db=40):
        """
        Parameters
        ----------
        noise0_db, noise1_db : float
            Absolute noise levels for E|wa|^2 and E|wb|^2
        sat_db : float
            Saturation level in dB relative to E|wa|^2
        """
        # Call Keras model intializer
        super().__init__(name='SatNL')

        # Set power levels
        self.var_wa = tf.constant(10**(0.1 * noise0_db))
        self.var_wb = tf.constant(10**(0.1 * noise1_db))
        self.Psat = tf.constant(10**(0.1 * sat_db) * self.var_wa)

    #@tf.function # Enable graph execution to speed things up
    def __call__(self, r):

        # Add the input noise
        u = add_awgn(r, self.var_wa)

        # Saturate the signal
        s = tf.abs(u) / np.sqrt(self.Psat)
        s_safe = tf.where(s > 0, s, tf.ones_like(s))
        f = tf.where(s > 0, tf.math.tanh(s_safe) / s_safe, tf.ones_like(s))
        ysat = real_to_complex(f) * u

        # Add the output noise
        y = add_awgn(ysat, self.var_wb)
       
        return y, f

    
class SatLinearEst(tf.keras.Layer): 
    """
    Linearized estimator for y = sat_nl(r)
    """
    def __init__(self, sat_nl):
        """
        Parameters
        ----------
        sat_nl : SatNL
            Nonlinearity saturation.
        """
        # Call Keras model intializer
        super().__init__(name='SatNLEst')

        # Save parameters
        self.var_wa = sat_nl.var_wa
        self.var_wb = sat_nl.var_wb
        self.Psat = sat_nl.Psat
    
    def call(self, 
            r_mean : tf.Tensor,
            r_var : tf.Tensor,
            y_obs : tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Given y = sat_nl(r) and r ~ CN(r_mean, r_var),
        computes an estimate for 
            r_est = E[r | y] and r_est_var = Var[r | y]

        The linear estimator models the output as:

            y ≈ f(r_mean) * (r + wₐ) + w_b
              = a * r + w̃

        where:
            - a = f(r_mean) is the saturation gain
            - w̃ = f(r_mean) * wₐ + w_b is the effective noise

        The linear MMSE estimator is then:

            gain = r_var / (|a|² * r_var + w_var)
            r_est ≈ r_mean + conj(a) * gain * (y - a * r_mean)
            w_var = |a|² * var(wₐ) + var(w_b)

        Parameters
        ----------
        r_mean : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Mean of the input signal r
        r_var : tf.Tensor, shape (nsamp, 1), tf.float32
            Variance of the input signal r (one value per sample block)
        y_obs : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Received signal after nonlinearity and noise

        Returns
        -------
        r_est : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Posterior mean estimate of the input signal r
        r_est_var : tf.Tensor, shape (nsamp, 1), tf.float32
            Posterior variance estimate of the input signal r
        """

        # Flatten the inputs and expand the dimension of the variance xvar0
        nsamp, ntd = r_mean.shape
        r_mean = tf.reshape(r_mean, (-1, 1))
        r_var = tf.reshape(tf.tile(r_var, (1, ntd)), (-1, 1))
        y_obs = tf.reshape(y_obs, (-1, 1))

        # Saturate the signal
        s = tf.abs(r_mean) / np.sqrt(self.Psat)
        s_safe = tf.where(s > 0, s, tf.ones_like(s))
        a = tf.where(s > 0, tf.math.tanh(s_safe) / s_safe, tf.ones_like(s))
        a = tf.cast(a, tf.complex64)
        
        # Compute the noise variance
        a_sq = tf.abs(a) ** 2
        w_var = a_sq * self.var_wa + self.var_wb

        # Compute the linear estimate
        gain = r_var / (a_sq * r_var + w_var)
        gain = tf.math.conj(a) * tf.cast(gain, tf.complex64)
        r_est = r_mean + gain * (y_obs - a * r_mean)
        r_var_post = r_var * w_var / (a_sq * r_var + w_var)

        # Reshape the outputs and average the variance of the output
        # over each sample block
        r_est = tf.reshape(r_est, (nsamp, ntd))
        r_var_post = tf.reshape(r_var_post, (nsamp, ntd))
        r_var_post = tf.reduce_mean(r_var_post, axis=1, keepdims=True)

        return r_est, r_var_post

    
class SatNeuralEst(tf.keras.Layer): 
    """
    Learned neural network estimator for y = sat_nl(r)
    """
    def __init__(self,
                 sat_nl: SatNL,
                 nhid: int = 64):
        """
        Parameters
        ----------
        sat_nl : SatNL
            Nonlinearity saturation.  
        nhid : int
            Number of hidden units in the neural network
        """
        # Call Keras model intializer
        super().__init__(name='SatNeuralEst')

        # Save parameters
        self.var_wa = sat_nl.var_wa
        self.var_wb = sat_nl.var_wb
        self.Psat = sat_nl.Psat

        # Neural network dimensions
        nout = 3 # [gain0, gain1, log_xvar]

        # Define the neural network layers
        self.dense1 = tf.keras.layers.Dense(nhid, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(nout, activation=None)

    def build(self, input_shape):
        """
        Build the neural network
        """
        # The input to dense1 is [z_1, gamma_1, y_obs]
        nin=3   

        dense1_in_shape = (nin,)
        self.dense1.build(dense1_in_shape)

        dense1_out_shape = self.dense1.compute_output_shape(dense1_in_shape)
        self.dense2.build(dense1_out_shape)
     
    def call(self, 
            z_1: tf.Tensor,
            gamma_1: tf.Tensor,
            y_obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        We follow a structure similar to the linear estimator,
        namely:

            r_est = z_1 + gain0 * (y_obs - gain1 * z_1)
            
        But we use a neural network to learn the gain0 and gain1
        parameters based on the input features:

            [gain0, gain1, log_r_est_var] = NeuralNet(z_1, gamma_1, y_obs)

        Parameters
        ----------
        z_1 : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Mean of the input signal r
        gamma_1 : tf.Tensor, shape (nsamp, 1), tf.float32
            Inverse variance of the input signal r (one value per sample block)
        y_obs : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Received signal y after the saturation nonlinearity and noise

        Returns
        -------
        v : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Posterior mean estimate of the input signal in time domain
        gamma_0 : tf.Tensor, shape (nsamp, 1), tf.float32
            Posterior inverse variance estimate of the input signal
        """

        # Get the dimensions
        nsamp, ntd = z_1.shape

        # Flatten the inputs
        z_1 = tf.reshape(z_1, (-1, 1))
        gamma_1 = tf.reshape(tf.tile(gamma_1, (1, ntd)), (-1, 1))
        y_obs = tf.reshape(y_obs, (-1, 1))
        
        # As features for the NN, we take the values normalized
        # by the saturation level
        r_mean_sat = tf.abs(z_1) / np.sqrt(self.Psat)
        r_var_sat = 1 / (gamma_1 * self.Psat)
        y_obs_sat = tf.abs(y_obs) / np.sqrt(self.Psat)
       
        # Column stack the inputs
        features = tf.concat([r_mean_sat, r_var_sat, y_obs_sat], axis=1)
       
        # Pass through the neural network to obtain the gains
        hidden = self.dense1(features)
        output = self.dense2(hidden)

        # Extract NN outputs
        gain0 = real_to_complex(tf.expand_dims(output[:, 0], axis=1))
        gain1 = real_to_complex(tf.expand_dims(output[:, 1], axis=1))
        log_r_var_post = tf.expand_dims(output[:, 2], axis=1)

        # Run the linear channel
        v = z_1 + gain0 * (y_obs - gain1 * z_1)

        # Get the log variance
        rho_1 =  gamma_1 / tf.exp(log_r_var_post)
   
        # Reshape the estimates
        v = tf.reshape(v, (nsamp, ntd))
        gamma_0 = 1 / tf.reduce_mean(tf.reshape(1 / rho_1, (nsamp, ntd)), axis=1, keepdims=True)
        
        return v, gamma_0


class SpecNeuralUpdate(tf.keras.Layer):
    """
    Learned neural network for updating message parameters and precision variables
    """

    def __init__(self,
                 sat_nl: SatNL,
                 nhid: int = 64):
        """
        Parameters
        ----------
        sat_nl : SatNL
            Nonlinearity saturation configuration, containing var_wa, var_wb, and Psat.
        nhid : int
            Number of hidden units in the neural network.
        """
        super().__init__(name='SpecNeuralUpdate')

        # Save saturation parameters
        self.var_wa = sat_nl.var_wa
        self.var_wb = sat_nl.var_wb
        self.Psat = sat_nl.Psat

        # Output size: [beta_0, beta_1, gamma_1]
        nout = 3

        # Define neural network layers
        self.dense1 = tf.keras.layers.Dense(nhid, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(nout, activation=None)

    def build(self, input_shape):
        """
        Build the layer weights based on the input shape.
        Expected input: concatenation of [z_0, gamma_0, S, mu].
        """
        nin = 4  # Number of features
        dense1_in_shape = (nin,)
        self.dense1.build(dense1_in_shape)

        dense1_out_shape = self.dense1.compute_output_shape(dense1_in_shape)
        self.dense2.build(dense1_out_shape)

    def call(self, 
             z_0: tf.Tensor,
             gamma_0: tf.Tensor,
             S: tf.Tensor,
             mu: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass through the neural network.

        Parameters
        ----------
        z_0 : tf.Tensor, shape (nsamp, ntd), tf.complex64
            Posterior mean of the input signal from the previous iteration.
        gamma_0 : tf.Tensor, shape (nsamp, 1), tf.float32
            Posterior precision (inverse variance) from the previous iteration.
        S : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior variance of the input signal.
        mu : tf.Tensor, shape (nsamp, ntd), tf.float32
            Prior mean of the input signal.

        Returns
        -------
        beta_0 : tf.Tensor, shape (nsamp, 1), tf.complex64
            Updated message coefficient.
        beta_1 : tf.Tensor, shape (nsamp, 1), tf.complex64
            Updated message coefficient.
        gamma_1 : tf.Tensor, shape (nsamp, 1), tf.float32
            Updated precision (inverse variance).
        """
        # Get input dimensions
        nsamp, ntd = z_0.shape

        # Flatten and prepare inputs for the neural network
        z_0 = tf.reshape(z_0, (-1, 1))  # (nsamp * ntd, 1)
        gamma_0 = tf.reshape(tf.tile(gamma_0, (1, ntd)), (-1, 1))  # Repeat gamma_0 across time dimension
        S = tf.reshape(S, (-1, 1))
        mu = tf.reshape(mu, (-1, 1))

        # As features for the NN, we take the values normalized
        # by the saturation level
        z_0 = tf.abs(z_0) / np.sqrt(self.Psat)
        r_var = 1 / (gamma_0 * self.Psat)
        S = S / self.Psat
        mu = tf.abs(mu) / np.sqrt(self.Psat)

        # Concatenate features: [z_0, gamma_0, S, mu]
        features = tf.concat([z_0, r_var, S, mu], axis=1)

        # Pass through the neural network
        hidden = self.dense1(features)
        output = self.dense2(hidden)

        # Extract outputs: real-valued [beta_0, beta_1, gamma_1]
        beta_0 = real_to_complex(tf.expand_dims(output[:, 0], axis=1))
        beta_1 = real_to_complex(tf.expand_dims(output[:, 1], axis=1))
        log_r_var_post = tf.expand_dims(output[:, 2], axis=1)
        rho_0 =  gamma_0 / tf.exp(log_r_var_post)

        # Reshape outputs to match expected output dimensions
        beta_0 = tf.reduce_mean(tf.reshape(beta_0, (nsamp, ntd)), axis=1, keepdims=True)
        beta_1 = tf.reduce_mean(tf.reshape(beta_1, (nsamp, ntd)), axis=1, keepdims=True)
        gamma_1 = 1 / tf.reduce_mean(tf.reshape(1 / rho_0, (nsamp, ntd)), axis=1, keepdims=True)

        return beta_0, beta_1, gamma_1