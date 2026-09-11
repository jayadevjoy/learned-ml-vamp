from plotter import Plotter

snr_values = [10, 20]
iters = [1, 2, 3]
train_steps = 2000
quantize = True

if quantize:
    for snr_filter in snr_values:
        for nitvamp in iters:
        
            plotter = Plotter(f"../results/data/vamp_quant_iter_{nitvamp}_epoch_{train_steps}.csv")

            plotter.custom_plot(
                metrics=['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"Achievable Rate (SNR = {snr_filter} dB, Iter = {nitvamp}, Quantized)",
                xlabel="INR (dB)",
                ylabel="Rate (bits/use)",
                save_path=f"../results/plots_quant/cap/cap_quant_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=False)
            
            plotter.custom_plot(
                metrics=['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"Normalized MSE (SNR = {snr_filter} dB, Iter = {nitvamp}, Quantized)",
                xlabel="INR (dB)",
                ylabel="Normalized MSE",
                save_path=f"../results/plots_quant/nmse/nmse_quant_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=True)
            
            plotter.custom_plot(
                metrics=['mse_vamp', 'mse_vamp_unk', 'mse_lin', 'mse_lin_unk', 'mse_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"MSE (SNR = {snr_filter} dB, Iter = {nitvamp}, Quantized)",
                xlabel="INR (dB)",
                ylabel="MSE",
                save_path=f"../results/plots_quant/mse/mse_quant_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=True)
else:
    for snr_filter in snr_values:
        for nitvamp in iters:
        
            plotter = Plotter(f"../results/data/vamp_iter_{nitvamp}_epoch_{train_steps}.csv")

            plotter.custom_plot(
                metrics=['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"Achievable Rate (SNR = {snr_filter} dB, Iter = {nitvamp})",
                xlabel="INR (dB)",
                ylabel="Rate (bits/use)",
                save_path=f"../results/plots/cap/cap_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=False)
            
            plotter.custom_plot(
                metrics=['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"Normalized MSE (SNR = {snr_filter} dB, Iter = {nitvamp})",
                xlabel="INR (dB)",
                ylabel="Normalized MSE",
                save_path=f"../results/plots/nmse/nmse_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=True)
            
            plotter.custom_plot(
                metrics=['mse_vamp', 'mse_vamp_unk', 'mse_lin', 'mse_lin_unk', 'mse_orc'],
                xaxis='inr',
                filter_by={'snr': snr_filter},
                title=f"MSE (SNR = {snr_filter} dB, Iter = {nitvamp})",
                xlabel="INR (dB)",
                ylabel="MSE",
                save_path=f"../results/plots/mse/mse_snr_{snr_filter}_iter_{nitvamp}_epoch_{train_steps}.pdf",
                log_y=True)