"""
Class for running all simulations and plotting results over a grid of SNR and INR values.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Import custom modules and components
from .utilities import metrics
from .vamp_sim import VampSim

class AllGridSim:

    def __init__(self, 
                 snr_list, 
                 inr_list, 
                 nfft=512, 
                 nsamp=1000, 
                 nitvamp=1, 
                 train_steps=500, 
                 retrain=True,  
                 des_known=True,
                 quantize=False):
        
        # ==== Simulation Config ====
        self.snr_list = snr_list
        self.inr_list = inr_list
        self.nfft = nfft
        self.nsamp = nsamp
        self.nitvamp = nitvamp
        self.train_steps = train_steps
        self.retrain = retrain
        self.des_known = des_known
        self.quantize = quantize

        self.results = []


    def run_sim(self):
        for snr in self.snr_list:
            for inr in self.inr_list:

                # ==== Run Simulation (Known Interferer) ====
                print(f"\nRunning simulation (Known Interferer) for SNR={snr} dB, INR={inr} dB")
                
                sim = VampSim(nfft=self.nfft, 
                               nitvamp=self.nitvamp,
                               nsamp=self.nsamp, 
                               snr=snr,
                               inr=inr,
                               intf_known=True,
                               quantize=self.quantize)

                if self.retrain:
                    sim.train(nsteps=self.train_steps)
                    load_model = False
                else:
                    load_model = True

                # ==== Evaluation ====
                r, r_hat_vamp, r_hat_lin, r_hat_orc = sim.evaluate(load_model=load_model)

                # ==== Compute Metrics ====
                mse_vamp, nmse_vamp = error_metric(r, r_hat_vamp, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                mse_lin, nmse_lin = error_metric(r, r_hat_lin, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                mse_orc, nmse_orc = error_metric(r, r_hat_orc, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                cap_vamp = corr_cap(r, r_hat_vamp, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                cap_lin = corr_cap(r, r_hat_lin, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                cap_orc = corr_cap(r, r_hat_orc, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)

                print(f"Capacity - VAMP (Known Interferer): {cap_vamp.numpy():.4f}")
                print(f"Capacity - Linear (Known Interferer): {cap_lin.numpy():.4f}")

                # # ==== SNR Evaluation ====
                # snr_vamp = sim.src.measure_snr(xtd, xvamp_est)
                # snr_lin = sim.src.measure_snr(xtd, xlin_est)
                # #snr_orc = sim.src.measure_snr(xtd, xorc_est)



                # ==== Run Simulation (Unknown Interferer) ====
                print(f"\nRunning simulation (Unknown Interferer) for SNR={snr} dB, INR={inr} dB")

                sim = VampSim(nfft=self.nfft, 
                               nitvamp=self.nitvamp,
                               nsamp=self.nsamp, 
                               snr=snr,
                               inr=inr,
                               intf_known=False,
                               quantize=self.quantize)

                if self.retrain:
                    sim.train(nsteps=self.train_steps)
                    load_model = False
                else:
                    load_model = True

                # ==== Evaluation ====
                r, r_hat_vamp, r_hat_lin, _ = sim.evaluate(load_model=load_model)

                # ==== Compute Metrics ====
                mse_vamp_unk, nmse_vamp_unk = error_metric(r, r_hat_vamp, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                mse_lin_unk, nmse_lin_unk = error_metric(r, r_hat_lin, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                cap_vamp_unk = corr_cap(r, r_hat_vamp, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)
                cap_lin_unk = corr_cap(r, r_hat_lin, des_known=self.des_known, des_idx0=sim.des_idx0, des_idx1=sim.des_idx1)

                print(f"Capacity - VAMP (Unknown Interferer): {cap_vamp_unk.numpy():.4f}")
                print(f"Capacity - Linear (Unknown Interferer): {cap_lin_unk.numpy():.4f}")

                # # ==== SNR Evaluation ====
                # snr_vamp_unk = sim.src.measure_snr(xtd, xvamp_est)
                # snr_lin_unk = sim.src.measure_snr(xtd, xlin_est)
                # snr_orc_unk = sim.src.measure_snr(xtd, xorc_est)

                self.results.append({
                    'snr': snr,
                    'inr': inr,
                    'cap_vamp': cap_vamp.numpy(),
                    'cap_lin': cap_lin.numpy(),
                    'cap_vamp_unk': cap_vamp_unk.numpy(),
                    'cap_lin_unk': cap_lin_unk.numpy(),
                    'cap_orc': cap_orc.numpy(),
                    'nmse_vamp': nmse_vamp.numpy(),
                    'nmse_lin': nmse_lin.numpy(),
                    'nmse_vamp_unk': nmse_vamp_unk.numpy(),
                    'nmse_lin_unk': nmse_lin_unk.numpy(),
                    'nmse_orc': nmse_orc.numpy(),
                    'mse_vamp': mse_vamp.numpy(),
                    'mse_lin': mse_lin.numpy(),
                    'mse_vamp_unk': mse_vamp_unk.numpy(),
                    'mse_lin_unk': mse_lin_unk.numpy(),
                    'mse_orc': mse_orc.numpy()})

    def save(self, file_path):
        """
        Save the simulation results to a CSV file.
        Args:
            file_path (str): Path to save the CSV file.
        """
        if not self.results:
            print("No results to save. Run simulations first.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")