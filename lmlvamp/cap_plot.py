import matplotlib.pyplot as plt
from plotter import Plotter

# Setup
snr_values = [10, 20]
iters = [1, 2, 3]
train_steps = 2000

# Create a 3x4 grid (rows=3, cols=4)
fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex='col')
fig.subplots_adjust(wspace=0.4, hspace=0.4)

for i, snr_filter in enumerate(snr_values):
    for j, nitvamp in enumerate(iters):
        row = j

        # === CAP (No Quantization) ===
        plotter_noq = Plotter(f"../results/data/vamp_iter_{nitvamp}_epoch_{train_steps}.csv")
        ax_cap_noq = axs[row, i]
        df_cap_noq = plotter_noq.df.copy()
        df_cap_noq = df_cap_noq[df_cap_noq['snr'] == snr_filter].sort_values('inr')

        for metric in ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']:
            label = plotter_noq.compact_labels.get(metric, metric)
            ax_cap_noq.plot(df_cap_noq['inr'], df_cap_noq[metric],
                            marker='o', label=label, linewidth=1, markersize=3)

        ax_cap_noq.grid(True, linewidth=0.3)
        if row == 2:
            ax_cap_noq.set_xlabel("INR (dB)", fontsize=9)
        ax_cap_noq.tick_params(labelsize=7)

        # === CAP (With Quantization) ===
        plotter_q = Plotter(f"../results/data/vamp_quant_iter_{nitvamp}_epoch_{train_steps}.csv")
        ax_cap_q = axs[row, i + 2]  # Shift by 2 columns
        df_cap_q = plotter_q.df.copy()
        df_cap_q = df_cap_q[df_cap_q['snr'] == snr_filter].sort_values('inr')

        for metric in ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']:
            label = plotter_q.compact_labels.get(metric, metric)
            ax_cap_q.plot(df_cap_q['inr'], df_cap_q[metric],
                          marker='o', label=label, linewidth=1, markersize=3)

        ax_cap_q.grid(True, linewidth=0.3)
        if row == 2:
            ax_cap_q.set_xlabel("INR (dB)", fontsize=9)
        ax_cap_q.tick_params(labelsize=7)

# === Legend ===
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 3].legend(handles, labels,
                 loc='upper right',
                 fontsize=9,
                 frameon=True)

# === Row labels (iteration titles) ===
row_titles = ['Iterations = 1', 'Iterations = 2', 'Iterations = 3']
for row in range(3):
    axs[row, 0].set_ylabel(row_titles[row], fontsize=11)

# === Column titles ===
column_titles = ['SNR = 10 dB', 'SNR = 20 dB', 'SNR = 10 dB', 'SNR = 20 dB']
for col in range(4):
    axs[0, col].set_title(column_titles[col], fontsize=11)

# === Shared block titles: "CAP" and "NMSE" ===
fig.text(0.268, 0.945, "Rate (bits/use) - Unquantized", ha='center', fontsize=13)
fig.text(0.762, 0.945, "Rate (bits/use) - Quantized", ha='center', fontsize=13)

# Final layout
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig("../results/plots_paper/cap_quant_vs_noquant.pdf", dpi=300)
plt.show()
