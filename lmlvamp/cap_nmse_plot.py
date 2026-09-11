import matplotlib.pyplot as plt
from plotter import Plotter

# Setup
snr_values = [10, 20]
iters = [1, 2, 3]
train_steps = 2000
quantize = False

# Create a 3x4 grid (rows=3, cols=4)
fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex='col')
fig.subplots_adjust(wspace=0.4, hspace=0.4)

if quantize:
    for i, snr_filter in enumerate(snr_values):
        for j, nitvamp in enumerate(iters):
            plotter = Plotter(f"../results/data/vamp_quant_iter_{nitvamp}_epoch_{train_steps}.csv")
            
            row = j
            col_cap = i
            ax_cap = axs[row, col_cap]
            df_cap = plotter.df.copy()
            df_cap = df_cap[df_cap['snr'] == snr_filter].sort_values('inr')

            for metric in ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']:
                label = plotter.compact_labels.get(metric, metric)
                ax_cap.plot(df_cap['inr'], df_cap[metric], marker='o', label=label, linewidth=1, markersize=3)

            ax_cap.grid(True, linewidth=0.3)
            if row == 2:
                ax_cap.set_xlabel("INR (dB)", fontsize=9)
            ax_cap.tick_params(labelsize=7)

            # NMSE subplot
            col_nmse = i + 2
            ax_nmse = axs[row, col_nmse]
            df_nmse = plotter.df.copy()
            df_nmse = df_nmse[df_nmse['snr'] == snr_filter].sort_values('inr')

            for metric in ['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc']:
                label = plotter.compact_labels.get(metric, metric)
                ax_nmse.semilogy(df_nmse['inr'], df_nmse[metric], marker='o', label=label, linewidth=1, markersize=3)

            ax_nmse.grid(True, which='both', linewidth=0.3)
            if row == 2:
                ax_nmse.set_xlabel("INR (dB)", fontsize=9)
            ax_nmse.tick_params(labelsize=7)

    # Get handles from one of the subplots (e.g., first CAP plot)
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Place a single legend in bottom-right subplot (axs[2, 3])
    axs[2, 3].legend(handles, labels,
                    loc='lower right',
                    fontsize=9,
                    frameon=True)

    # === Set row labels (iteration titles) on the left side ===
    row_titles = ['Iterations = 1', 'Iterations = 2', 'Iterations = 3']
    for row in range(3):
        axs[row, 0].set_ylabel(row_titles[row], fontsize=11)

    # === Shared column subheadings: SNR titles above each column ===
    column_titles = ['SNR = 10 dB', 'SNR = 20 dB', 'SNR = 10 dB', 'SNR = 20 dB']
    for col in range(4):
        axs[0, col].set_title(column_titles[col], fontsize=11)

    # === Shared block titles: "CAP" and "NMSE" ===
    fig.text(0.268, 0.945, "Rate (bits/use) - With Quantization", ha='center', fontsize=13)
    fig.text(0.762, 0.945, "Normalized MSE - With Quantization", ha='center', fontsize=13)

    # Final layout and save (no global suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig("../results/plots_paper/cap_nmse_quant_grid.pdf", dpi=300)
    plt.show()
else:
    for i, snr_filter in enumerate(snr_values):
        for j, nitvamp in enumerate(iters):
            plotter = Plotter(f"../results/data/vamp_iter_{nitvamp}_epoch_{train_steps}.csv")
            
            row = j
            col_cap = i
            ax_cap = axs[row, col_cap]
            df_cap = plotter.df.copy()
            df_cap = df_cap[df_cap['snr'] == snr_filter].sort_values('inr')

            for metric in ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']:
                label = plotter.compact_labels.get(metric, metric)
                ax_cap.plot(df_cap['inr'], df_cap[metric], marker='o', label=label, linewidth=1, markersize=3)

            ax_cap.grid(True, linewidth=0.3)
            if row == 2:
                ax_cap.set_xlabel("INR (dB)", fontsize=9)
            ax_cap.tick_params(labelsize=7)

            # NMSE subplot
            col_nmse = i + 2
            ax_nmse = axs[row, col_nmse]
            df_nmse = plotter.df.copy()
            df_nmse = df_nmse[df_nmse['snr'] == snr_filter].sort_values('inr')

            for metric in ['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc']:
                label = plotter.compact_labels.get(metric, metric)
                ax_nmse.semilogy(df_nmse['inr'], df_nmse[metric], marker='o', label=label, linewidth=1, markersize=3)

            ax_nmse.grid(True, which='both', linewidth=0.3)
            if row == 2:
                ax_nmse.set_xlabel("INR (dB)", fontsize=9)
            ax_nmse.tick_params(labelsize=7)

    # Get handles from one of the subplots (e.g., first CAP plot)
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Place a single legend in bottom-right subplot (axs[2, 3])
    axs[2, 3].legend(handles, labels,
                    loc='lower right',
                    fontsize=9,
                    frameon=True)

    # === Set row labels (iteration titles) on the left side ===
    row_titles = ['Iterations = 1', 'Iterations = 2', 'Iterations = 3']
    for row in range(3):
        axs[row, 0].set_ylabel(row_titles[row], fontsize=11)

    # === Shared column subheadings: SNR titles above each column ===
    column_titles = ['SNR = 10 dB', 'SNR = 20 dB', 'SNR = 10 dB', 'SNR = 20 dB']
    for col in range(4):
        axs[0, col].set_title(column_titles[col], fontsize=11)

    # === Shared block titles: "CAP" and "NMSE" ===
    fig.text(0.268, 0.945, "Rate (bits/use) - No Quantization", ha='center', fontsize=13)
    fig.text(0.762, 0.945, "Normalized MSE - No Quantization", ha='center', fontsize=13)

    # Final layout and save (no global suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig("../results/plots_paper/cap_nmse_grid.pdf", dpi=300)
    plt.show()