"""
Generating results and plots for all simulations
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Dynamically reload custom modules (for interactive environments like Jupyter)
import importlib
modules_to_reload = ['complex_utils', 'nonlin', 'specsource', 'learned_vamp', 'vamp_sim', 'all_sim']
for mod_name in modules_to_reload:
    try:
        module = importlib.reload(importlib.import_module(mod_name))
        print(f"Reloaded module: {mod_name}")
    except Exception as e:
        module = importlib.import_module(mod_name)
        print(f"Imported module: {mod_name}")

# Import custom modules and components
from all_sim import AllGridSim

# ==== Simulation Config ====
nfft = 512
nsamp = 1000
nitvamp = 2
retrain = True
train_steps = 2000  # Number of training steps for VAMP estimator
des_known = True  # Set to True if the desired signal bandwidth is known
quantize = True  # Set to True if quantization is applied
# intf_known = True  # Set to True if the interferer is known to the receiver

# Define a grid of SNR values
snr_values = np.arange(10, 21, 10)  # Desired SNR (SNR)
inr_values = np.arange(30, 81, 5)   # Interferer SNR (INR)
# snr_values = [10]   # Desired SNR (SNR)

simplot = AllGridSim(snr_list=snr_values,
                     inr_list=inr_values,
                     nfft=nfft, 
                     nsamp=nsamp,
                     nitvamp=nitvamp,
                     train_steps=train_steps,
                     retrain=retrain,
                     des_known=des_known,
                     quantize=quantize)

# Run all simulations
simplot.run_sim()

# Save results to a CSV file
if quantize:
    simplot.save(f"../results/data/vamp_quant_iter_{nitvamp}_epoch_{train_steps}.csv")
else:
    simplot.save(f"../results/data/vamp_iter_{nitvamp}_epoch_{train_steps}.csv")

# # Save results to a CSV file
# if quantize:
#     simplot.save(f"../results/data/vamp_quant_snr_{snr_values[0]}_iter_{nitvamp}_epoch_{train_steps}.csv")
# else:
#     simplot.save(f"../results/data/vamp_snr_{snr_values[0]}_iter_{nitvamp}_epoch_{train_steps}.csv")