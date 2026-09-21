"""
Class for running all simulations and plotting results over a grid of SNR and INR values.
"""

import tensorflow as tf
import pandas as pd

# Import custom modules and components
from .utilities import metrics
from .channel_sim import VampSim

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
                 neural_update=True,
                 quantize=False):
        
        self.snr_list = snr_list
        self.inr_list = inr_list
        self.nfft = nfft
        self.nsamp = nsamp
        self.nitvamp = nitvamp
        self.train_steps = train_steps
        self.retrain = retrain
        self.des_known = des_known
        self.neural_update = neural_update
        self.quantize = quantize

        self.results = []


    def run_sim(self):
        for snr in self.snr_list:
            for inr in self.inr_list:

                result_row = {'snr': snr, 'inr': inr}
                mask = None

                for intf_known, suffix, label in [(False, '_unk', 'Unknown Interferer'),
                                                    (True,  '',    'Known Interferer')]:

                    print(f"\nRunning simulation ({label}) for SNR={snr} dB, INR={inr} dB")

                    sim = VampSim(nfft=self.nfft,
                                  nitvamp=self.nitvamp,
                                  nsamp=self.nsamp,
                                  snr=snr,
                                  inr=inr,
                                  intf_known=intf_known,
                                  neural_update=self.neural_update,
                                  quantize=self.quantize)

                    if self.retrain:
                        sim.train(nsteps=self.train_steps)
                        load_model = False
                    else:
                        load_model = True

                    # Evaluation
                    r, r_hat_vamp, r_hat_lin, r_hat_orc = sim.evaluate(load_model=load_model)

                    # Build mask once
                    if mask is None:
                        mask_1d = tf.concat([tf.zeros(sim.des_idx0, dtype=tf.complex64),
                                            tf.ones(sim.des_idx1 - sim.des_idx0, dtype=tf.complex64),
                                            tf.zeros(self.nfft - sim.des_idx1, dtype=tf.complex64)], axis=0)
                        mask = tf.broadcast_to(mask_1d, tf.shape(r))

                    # Compute Metrics
                    cap_vamp, mse_vamp, nmse_vamp = metrics(r, r_hat_vamp, des_known=self.des_known, mask=mask)
                    cap_lin, mse_lin, nmse_lin = metrics(r, r_hat_lin, des_known=self.des_known, mask=mask)

                    print(f"Capacity - VAMP ({label}): {cap_vamp.numpy():.4f}")
                    print(f"Capacity - Linear ({label}): {cap_lin.numpy():.4f}")

                    result_row[f'cap_vamp{suffix}'] = cap_vamp.numpy()
                    result_row[f'cap_lin{suffix}'] = cap_lin.numpy()
                    result_row[f'nmse_vamp{suffix}'] = nmse_vamp.numpy()
                    result_row[f'nmse_lin{suffix}'] = nmse_lin.numpy()
                    result_row[f'mse_vamp{suffix}'] = mse_vamp.numpy()
                    result_row[f'mse_lin{suffix}'] = mse_lin.numpy()

                    # Oracle needs to be computed only once
                    if intf_known:
                        cap_orc, mse_orc, nmse_orc = metrics(r, r_hat_orc, des_known=self.des_known, mask=mask)
                        result_row['cap_orc'] = cap_orc.numpy()
                        result_row['nmse_orc'] = nmse_orc.numpy()
                        result_row['mse_orc'] = mse_orc.numpy()

                    # # SNR Evaluation
                    # snr_vamp{suffix} = sim.src.measure_snr(xtd, xvamp_est)
                    # snr_lin{suffix} = sim.src.measure_snr(xtd, xlin_est)
                    # snr_orc = sim.src.measure_snr(xtd, xorc_est)

                self.results.append(result_row)


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