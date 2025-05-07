import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def plot_custom(df_clean, x_col, y_col, xlabel, ylabel, title, filename, output_dir, model_color_map, algo_marker_map, hardware_line_map, log=False):
    plt.figure(figsize=(9, 6))
    grouped = df_clean.groupby(['Model', 'Algo', 'Hardware'])

    legend_handles = {}

    for (model, algo, hw), group in grouped:
        group_sorted = group.sort_values('Sampling')
        color = model_color_map.get(model, 'black')
        marker = algo_marker_map.get(algo, 'x')
        linestyle = hardware_line_map.get(hw, '-')
        label = f"{model}, {algo}, {hw}"

        # Plot individual points
        for i, (_, row) in enumerate(group_sorted.iterrows()):
            plt.plot(row[x_col], row[y_col], marker=marker, color=color,
                     linestyle='None', label=label if i == 0 else None)

        # Plot connecting line
        plt.plot(group_sorted[x_col], group_sorted[y_col], linestyle=linestyle,
                 color=color, alpha=0.7)

        # Create marker-only and line-only handles for legend
        if label not in legend_handles:
            marker_handle = Line2D([0], [0], marker=marker, linestyle='None',
                                   color=color, markersize=6)
            line_handle = Line2D([0], [0], linestyle=linestyle,
                                 color=color, linewidth=1.5)
            legend_handles[label] = (marker_handle, line_handle)

    # Add the combined marker-line legend using HandlerTuple
    plt.legend(legend_handles.values(), legend_handles.keys(),
               handler_map={tuple: HandlerTuple(ndivide=None)},
               bbox_to_anchor=(1.0, 0.5), loc='center left',
               fontsize='x-small', frameon=False)

    # Plot baseline lines if y-axis is Accuracy
    if 'Accuracy' in ylabel:
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()

        # LLaMA-1B-Instruct baseline
        plt.axhline(y=29, color='gray', linestyle='--', linewidth=1.2)
        plt.text(x=xmax * 0.99, y=29 + 0.5, s='LLaMA-1B-Instruct (29%)',
                 va='bottom', ha='right', fontsize='x-small', color='black')

        # LLaMA-8B-Instruct baseline
        plt.axhline(y=52, color='black', linestyle='--', linewidth=1.2)
        plt.text(x=xmax * 0.99, y=52 + 0.5, s='LLaMA-8B-Instruct (52%)',
                 va='bottom', ha='right', fontsize='x-small', color='black')

    if log:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both' if log else 'major', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join("plots", csv_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean CSV
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=['Latency (h)', 'Total Energy (kWh)', 'Accuracy']).copy()

    # Compute derived metrics
    df_clean['Latency-Energy Product'] = df_clean['Latency (h)'] * df_clean['Total Energy (kWh)']
    df_clean['Accuracy/Energy'] = df_clean['Accuracy'] * df_clean['Total Energy (kWh)']
    df_clean['Accuracy/Latency'] = df_clean['Accuracy'] * df_clean['Latency (h)']

    # Styling maps
    unique_models = sorted(df_clean['Model'].unique())
    fixed_colors = ['#E97132', '#156082', '#196B24']  # orange, blue, green

    model_color_map = {
        model: fixed_colors[i % len(fixed_colors)]
        for i, model in enumerate(unique_models)
    }

    algo_marker_map = {
        'best of n': 'o',
        'dvts': '^',
        'beam search': 's'
    }

    hardware_line_map = {
        'A100': '-.',
        'H200': ':',
        'A40': '--'
    }

    # Plot calls
    plot_custom(df_clean, 'Latency (h)', 'Accuracy', 'Latency (h)', 'Accuracy (%)',
                '1. Accuracy vs Latency', '1_accuracy_vs_latency.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    plot_custom(df_clean, 'Total Energy (kWh)', 'Accuracy', 'Total Energy (kWh)', 'Accuracy (%)',
                '2. Accuracy vs Energy', '2_accuracy_vs_energy.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    plot_custom(df_clean, 'Sampling', 'Accuracy', 'Sampling (n)', 'Accuracy (%)',
                '3. Accuracy vs Sampling (n)', '3_accuracy_vs_sampling.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    plot_custom(df_clean, 'Accuracy', 'Latency-Energy Product', 'Accuracy (%)', 'Latency × Energy (kWh·h)',
                '4a. Accuracy vs Latency-Energy Product', '4a_accuracy_vs_latency_energy_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    plot_custom(df_clean, 'Latency (h)', 'Accuracy/Energy', 'Latency (h)', 'Accuracy x Energy',
                '5a. Latency vs Accuracy-Energy Product', '5a_latency_vs_acc_energy_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    plot_custom(df_clean, 'Total Energy (kWh)', 'Accuracy/Latency', 'Total Energy (kWh)', 'Accuracy x Latency',
                '6a. Energy vs Accuracy-Latency Product', '6a_energy_vs_acc_latency_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map)

    # Log scale plots
    plot_custom(df_clean, 'Accuracy', 'Latency-Energy Product', 'Accuracy (%)', 'Latency × Energy (kWh·h)',
                '4b. Accuracy vs Latency-Energy Product Log Scale', '4b_accuracy_vs_latency_energy_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map, log=True)

    plot_custom(df_clean, 'Latency (h)', 'Accuracy/Energy', 'Latency (h)', 'Accuracy x Energy',
                '5b. Latency vs Accuracy-Energy Product Log Scale', '5b_latency_vs_acc_energy_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map, log=True)

    plot_custom(df_clean, 'Total Energy (kWh)', 'Accuracy/Latency', 'Total Energy (kWh)', 'Accuracy x Latency',
                '6b. Energy vs Accuracy-Latency Product Log Scale', '6b_energy_vs_acc_latency_product.png', output_dir,
                model_color_map, algo_marker_map, hardware_line_map, log=True)

    print(f"✅ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
