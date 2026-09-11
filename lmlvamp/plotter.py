import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, csv_path):
        """
        Initialize the Plotter with data from a CSV file.
        """
        self.df = pd.read_csv(csv_path)

        # Define default compact labels for known metric names
        self.compact_labels = {
            'cap_vamp': 'LMLVAMP-K',
            'cap_lin': 'LINEAR-K',
            'cap_vamp_unk': 'LMLVAMP-U',
            'cap_lin_unk': 'LINEAR-U',
            'cap_orc': 'ORACLE',
            'nmse_vamp': 'LMLVAMP-K',
            'nmse_lin': 'LINEAR-K',
            'nmse_vamp_unk': 'LMLVAMP-U',
            'nmse_lin_unk': 'LINEAR-U',
            'nmse_orc': 'ORACLE',
            'mse_vamp': 'LMLVAMP-K',
            'mse_lin': 'LINEAR-K',
            'mse_vamp_unk': 'LMLVAMP-U',
            'mse_lin_unk': 'LINEAR-U',
            'mse_orc': 'ORACLE'}

    def custom_plot(self, 
                    metrics, 
                    xaxis='inr', 
                    filter_by=None, 
                    title="Plot of Metrics vs INR", 
                    xlabel="INR (dB)",
                    ylabel="Metric Value",
                    save_path=None, 
                    log_y=False,
                    custom_label=None,
                    figsize=(3, 2.5),
                    fontsize=8):
        """
        Plot multiple metrics against a selected axis (snr or inr) on the same graph.

        Args:
            metrics (list of str): Metric names to plot (e.g., ['nmse_vamp', 'cap_vamp_unk']).
            xaxis (str): X-axis variable (e.g., 'snr' or 'inr').
            filter_by (dict): Filter conditions like {'snr': 10}.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            save_path (str): File path to save the plot (optional).
            log_y (bool): Use log scale on Y-axis.
            custom_label (list of str): Custom legend labels.
            figsize (tuple): Plot size (in inches).
            fontsize (int): Font size for labels and ticks.
        """
        df = self.df.copy()

        # Apply filters
        if filter_by:
            for key, val in filter_by.items():
                df = df[df[key] == val]

        if df.empty:
            print("No data matches the filtering criteria.")
            return

        df_sorted = df.sort_values(by=xaxis)

        plt.figure(figsize=figsize)

        for idx, metric in enumerate(metrics):
            if metric not in df_sorted.columns:
                print(f"Warning: '{metric}' not found in data.")
                continue
            
            if custom_label and idx < len(custom_label):
                label = custom_label[idx]
            else:
                label = self.compact_labels.get(metric, metric)

            if log_y:
                plt.semilogy(df_sorted[xaxis], df_sorted[metric], marker='o', ms=3, label=label, linewidth=1)
            else:
                plt.plot(df_sorted[xaxis], df_sorted[metric], marker='o', ms=3, label=label, linewidth=1)

        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.xticks(fontsize=fontsize - 1)
        plt.yticks(fontsize=fontsize - 1)
        plt.grid(True, which='both' if log_y else 'major', linewidth=0.3)
        plt.legend(fontsize=fontsize - 2)
        plt.tight_layout(pad=0.2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()