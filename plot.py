import matplotlib.pyplot as plt
from lmlvamp import Plotter

# Setup
snr_values = [5, 10]
iters = [1, 2, 3]
train_steps = 1000
quantize = True

# Tags
quant_tag = "quant" if quantize else "unquant"

data_prefix_fixed = f"vamp_{quant_tag}_fixed"
data_prefix_neural = f"vamp_{quant_tag}_neural"

outfile = f"results/plots/cap_nmse_{quant_tag}_overlay_grid.pdf"

quant_title = "Quantized" if quantize else "Unquantized"

cap_metrics = ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']
nmse_metrics = ['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc']

variant_metrics = {'cap_vamp', 'cap_vamp_unk', 'nmse_vamp', 'nmse_vamp_unk'}

cap_overlay_metrics = ['cap_vamp', 'cap_vamp_unk']
nmse_overlay_metrics = ['nmse_vamp', 'nmse_vamp_unk']

# Unified style dictionary
STYLES = {
    'lin':      dict(color='tab:green',  linestyle='-',  marker='o'),
    'lin_unk':  dict(color='tab:green',  linestyle='--', marker='D'),
    'orc':      dict(color='tab:gray',   linestyle='-',  marker='x'),
    'vamp_variant':     dict(color='tab:blue', linestyle='-',  marker='o'),
    'vamp_unk_variant': dict(color='tab:blue', linestyle='--', marker='D'),
    'vamp_neural':      dict(color='tab:red', linestyle='-',  marker='o'),
    'vamp_unk_neural':  dict(color='tab:red', linestyle='--', marker='D'),
}

def base_key(metric):
    """Strip 'cap_'/'nmse_' prefix -> e.g. 'cap_lin_unk' -> 'lin_unk'."""
    return metric.split('_', 1)[1]

# Figure
fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex='col')
fig.subplots_adjust(wspace=0.4, hspace=0.4)

for i, snr in enumerate(snr_values):
    for row, nitvamp in enumerate(iters):

        plotter_fixed = Plotter(
            f"results/data/{data_prefix_fixed}_iter_{nitvamp}_epoch_{train_steps}.csv"
        )
        df_fixed = plotter_fixed.df
        df_fixed = df_fixed[df_fixed['snr'] == snr].sort_values('inr')

        plotter_neural = Plotter(
            f"results/data/{data_prefix_neural}_iter_{nitvamp}_epoch_{train_steps}.csv"
        )
        df_neural = plotter_neural.df
        df_neural = df_neural[df_neural['snr'] == snr].sort_values('inr')

        # CAP
        ax = axs[row, i]
        for metric in cap_metrics:
            key = base_key(metric)
            if metric in variant_metrics:
                key += '_variant'
            style = STYLES[key]
            label = plotter_fixed.compact_labels.get(metric, metric)
            if metric in variant_metrics:
                label += ' (Variant)'
            ax.plot(
                df_fixed['inr'], df_fixed[metric],
                label=label, linewidth=1.2, markersize=4,
                markerfacecolor='white' if 'unk' in metric else style['color'],
                **style
            )
        for metric in cap_overlay_metrics:
            key = base_key(metric) + '_neural'
            style = STYLES[key]
            label = plotter_neural.compact_labels.get(metric, metric)
            ax.plot(
                df_neural['inr'], df_neural[metric],
                label=label, linewidth=1.2, markersize=4,
                markerfacecolor='white' if 'unk' in metric else style['color'],
                **style
            )

        # NMSE
        ax = axs[row, i + 2]
        for metric in nmse_metrics:
            key = base_key(metric)
            if metric in variant_metrics:
                key += '_variant'
            style = STYLES[key]
            label = plotter_fixed.compact_labels.get(metric, metric)
            if metric in variant_metrics:
                label += ' (Variant)'
            ax.semilogy(
                df_fixed['inr'], df_fixed[metric],
                label=label, linewidth=1.2, markersize=4,
                markerfacecolor='white' if 'unk' in metric else style['color'],
                **style
            )
        for metric in nmse_overlay_metrics:
            key = base_key(metric) + '_neural'
            style = STYLES[key]
            label = plotter_neural.compact_labels.get(metric, metric)
            ax.semilogy(
                df_neural['inr'], df_neural[metric],
                label=label, linewidth=1.2, markersize=4,
                markerfacecolor='white' if 'unk' in metric else style['color'],
                **style
            )

# Formatting
for row in range(3):
    axs[row, 0].set_ylabel(f"Iterations = {iters[row]}", fontsize=11)

for col in range(4):
    axs[2, col].set_xlabel("INR (dB)", fontsize=9)
    axs[0, col].set_title(f"SNR = {snr_values[col % 2]} dB", fontsize=11)
    for row in range(3):
        axs[row, col].grid(True, which='both' if col >= 2 else 'major', linewidth=0.3)
        axs[row, col].tick_params(labelsize=7)

fig.text(0.268, 0.945, f"Rate (bits/use) - {quant_title}", ha='center', fontsize=13)
fig.text(0.762, 0.945, f"Normalized MSE - {quant_title}", ha='center', fontsize=13)

handles, labels = axs[0, 0].get_legend_handles_labels()
axs[2, 3].legend(handles, labels, loc='lower right', fontsize=8, frameon=True, ncol=1)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig(outfile, dpi=300)
plt.show()

# # Setup
# snr_values = [5, 10]
# iters = [1, 2, 3]
# train_steps = 1000
# neural_update = True
# quantize = False

# # Tags
# quant_tag = "quant" if quantize else "unquant"
# update_tag = "neural" if neural_update else "fixed"

# data_prefix = f"vamp_{quant_tag}_{update_tag}"
# outfile = f"results/plots/cap_nmse_{quant_tag}_{update_tag}_grid.pdf"

# quant_title = "Quantized" if quantize else "Unquantized"
# update_title = "Neural Update" if neural_update else "Fixed Update"

# # Metrics
# cap_metrics = ['cap_vamp', 'cap_vamp_unk', 'cap_lin', 'cap_lin_unk', 'cap_orc']
# nmse_metrics = ['nmse_vamp', 'nmse_vamp_unk', 'nmse_lin', 'nmse_lin_unk', 'nmse_orc']

# # Figure
# fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex='col')
# fig.subplots_adjust(wspace=0.4, hspace=0.4)

# for i, snr in enumerate(snr_values):
#     for row, nitvamp in enumerate(iters):

#         plotter = Plotter(
#             f"results/data/{data_prefix}_iter_{nitvamp}_epoch_{train_steps}.csv"
#         )

#         df = plotter.df
#         df = df[df['snr'] == snr].sort_values('inr')

#         # CAP
#         ax = axs[row, i]
#         for metric in cap_metrics:
#             ax.plot(
#                 df['inr'], df[metric],
#                 marker='o',
#                 label=plotter.compact_labels.get(metric, metric),
#                 linewidth=1,
#                 markersize=3
#             )

#         # NMSE
#         ax = axs[row, i + 2]
#         for metric in nmse_metrics:
#             ax.semilogy(
#                 df['inr'], df[metric],
#                 marker='o',
#                 label=plotter.compact_labels.get(metric, metric),
#                 linewidth=1,
#                 markersize=3
#             )

# # Formatting
# for row in range(3):
#     axs[row, 0].set_ylabel(f"Iterations = {iters[row]}", fontsize=11)

# for col in range(4):
#     axs[2, col].set_xlabel("INR (dB)", fontsize=9)
#     axs[0, col].set_title(
#         f"SNR = {snr_values[col % 2]} dB",
#         fontsize=11
#     )

#     for row in range(3):
#         axs[row, col].grid(
#             True,
#             which='both' if col >= 2 else 'major',
#             linewidth=0.3
#         )
#         axs[row, col].tick_params(labelsize=7)

# # Block titles
# fig.text(
#     0.268, 0.945,
#     f"Rate (bits/use) - {quant_title}",
#     ha='center', fontsize=13
# )
# fig.text(
#     0.762, 0.945,
#     f"Normalized MSE - {quant_title}",
#     ha='center', fontsize=13
# )

# # # Optional update indicator
# # fig.text(
# #     0.5, 0.975,
# #     update_title,
# #     ha='center', fontsize=13
# # )

# # Single legend
# handles, labels = axs[0, 0].get_legend_handles_labels()
# axs[2, 3].legend(
#     handles, labels,
#     loc='lower right',
#     fontsize=9,
#     frameon=True
# )

# plt.tight_layout(rect=[0, 0.03, 1, 0.94])
# plt.savefig(outfile, dpi=300)
# plt.show()