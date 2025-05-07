import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# Global variables
df_clean = None
output_dir = None
model_color_map = {}
algo_marker_map = {}
hardware_line_map = {}

def plot_custom(x_col, y_col, xlabel, ylabel, title, filename, logx=False, logy=False, log_base=10):
    global df_clean, output_dir, model_color_map, algo_marker_map, hardware_line_map

    subfolder = "log" if logx or logy else "base"
    plot_dir = os.path.join(output_dir, subfolder)
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    grouped = df_clean.groupby(['Model', 'Algo', 'Hardware'])

    for (model, algo, hw), group in grouped:
        group_sorted = group.sort_values('Sampling')
        color = model_color_map.get(model, 'black')
        marker = algo_marker_map.get(algo, 'x')
        linestyle = hardware_line_map.get(hw, '-')
        for i, (_, row) in enumerate(group_sorted.iterrows()):
            ax.plot(row[x_col], row[y_col], marker=marker, color=color, linestyle='None')
        ax.plot(group_sorted[x_col], group_sorted[y_col], linestyle=linestyle, color=color, alpha=0.7)

    if ylabel.strip() == 'Accuracy (%)':
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.axhline(y=29, color='black', linestyle='--', linewidth=1.0)
        ax.text(x=xmax * 0.99, y=29 + 0.5, s='LLaMA-1B-Instruct (29%)', va='bottom', ha='right', fontsize='x-small', color='black')
        ax.axhline(y=52, color='black', linestyle='--', linewidth=1.0)
        ax.text(x=xmax * 0.99, y=52 + 0.5, s='LLaMA-8B-Instruct (52%)', va='bottom', ha='right', fontsize='x-small', color='black')

    if logx:
        ax.set_xscale('log', base=log_base)
    if logy:
        ax.set_yscale('log', base=log_base)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both' if logx or logy else 'major', linestyle='--', linewidth=0.5)

    fig.subplots_adjust(right=0.75)
    fig.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def render_legends():
    global output_dir, model_color_map, algo_marker_map, hardware_line_map, df_clean
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    legend_handles = {}
    grouped = df_clean.groupby(['Model', 'Algo', 'Hardware'])
    for (model, algo, hw), group in grouped:
        color = model_color_map.get(model, 'black')
        marker = algo_marker_map.get(algo, 'x')
        linestyle = hardware_line_map.get(hw, '-')
        label = f"{model}, {algo}, {hw}"
        if label not in legend_handles:
            marker_handle = Line2D([0], [0], marker=marker, linestyle='None', color=color, markersize=5)
            line_handle = Line2D([0], [0], linestyle=linestyle, color=color, linewidth=1.0)
            legend_handles[label] = (marker_handle, line_handle)

    legend_elements = []
    legend_labels = []
    legend_elements.append(Line2D([], [], linestyle='None', label='Marker = Algo', color='black'))
    legend_labels.append(r'$bf{Marker = Algo}$')
    legend_elements[-1] = Line2D([], [], linestyle='None', label=r'$bf{Marker = Algo}$', color='black', linewidth=0, markersize=0)
    for algo, marker in algo_marker_map.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=5))
        legend_labels.append(f"{algo}")

    legend_elements.append(Line2D([], [], linestyle='None', label='Color = Model', color='black'))
    legend_labels.append(r'$bf{Color = Model}$')
    legend_elements[-1] = Line2D([], [], linestyle='None', label=r'$bf{Color = Model}$', color='black', linewidth=0, markersize=0)
    for model, color in model_color_map.items():
        legend_elements.append(Line2D([0], [0], color=color, linestyle='-', linewidth=1.0))
        legend_labels.append(f"{model}")

    legend_elements.append(Line2D([], [], linestyle='None', label='Line = Hardware', color='black'))
    legend_labels.append(r'$bf{Line = Hardware}$')
    legend_elements[-1] = Line2D([], [], linestyle='None', label=r'$bf{Line = Hardware}$', color='black', linewidth=0, markersize=0)
    for hw, style in hardware_line_map.items():
        legend_elements.append(Line2D([0], [0], color='black', linestyle=style, linewidth=1.0))
        legend_labels.append(f"{hw}")

    legend_elements.append(Line2D([], [], linestyle='None', label='Model, Algo, Hardware', color='black'))
    legend_labels.append(r'$bf{Model, Algo, Hardware}$')
    legend_elements[-1] = Line2D([], [], linestyle='None', label=r'$bf{Model, Algo, Hardware}$', color='black', linewidth=0, markersize=0)
    for label, (marker_handle, line_handle) in legend_handles.items():
        combined_handle = (Line2D([0], [0], marker=marker_handle.get_marker(), linestyle='None', color=line_handle.get_color(), markersize=5),
                           Line2D([0], [0], linestyle=line_handle.get_linestyle(), color=line_handle.get_color(), linewidth=1.0))
        legend_elements.append(combined_handle)
        legend_labels.append(label)

    fig_leg, ax_leg = plt.subplots(figsize=(2, 2.65))
    ax_leg.set_title('Model, Algo, Hardware Legend', fontsize='medium')
    fig_leg.legend(legend_elements[-len(legend_handles):], legend_labels[-len(legend_handles):],
                   handler_map={tuple: HandlerTuple(ndivide=None)},
                   loc='center', fontsize='small', frameon=False)
    ax_leg.axis('off')
    fig_leg.tight_layout()
    fig_leg.savefig(os.path.join(output_dir, 'legend_combined.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_leg)

    legend_elements_simple = []
    legend_labels_simple = []

    for algo, marker in algo_marker_map.items():
        legend_elements_simple.append(Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=5))
        legend_labels_simple.append(f"Marker = {algo}")

    for model, color in model_color_map.items():
        legend_elements_simple.append(Line2D([0], [0], color=color, linestyle='-', linewidth=1.0))
        legend_labels_simple.append(f"Color = {model}")

    for hw, style in hardware_line_map.items():
        legend_elements_simple.append(Line2D([0], [0], color='black', linestyle=style, linewidth=1.0))
        legend_labels_simple.append(f"Line = {hw}")

    fig_legend2, ax_legend2 = plt.subplots(figsize=(2, 2))
    ax_legend2.set_title('Legend Key: Marker, Color, Line', fontsize='medium')
    fig_legend2.legend(legend_elements_simple, legend_labels_simple,
                       loc='center', fontsize='small', frameon=False)
    ax_legend2.axis('off')
    fig_legend2.tight_layout()
    fig_legend2.savefig(os.path.join(output_dir, 'legend_key.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_legend2)

def main():
    global df_clean, output_dir, model_color_map, algo_marker_map, hardware_line_map

    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = "plots3"
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
    model_color_map = {model: fixed_colors[i % len(fixed_colors)] for i, model in enumerate(unique_models)}

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

    # Standard plots
    plot_custom('Latency (h)', 'Accuracy', 'Latency (h)', 'Accuracy (%)',
                'Accuracy vs Latency', '1_accuracy_vs_latency.png')

    plot_custom('Total Energy (kWh)', 'Accuracy', 'Total Energy (kWh)', 'Accuracy (%)',
                'Accuracy vs Energy', '2_accuracy_vs_energy.png')

    plot_custom('Sampling', 'Accuracy', 'Sampling (n)', 'Accuracy (%)',
                'Accuracy vs Sampling (n)', '3_accuracy_vs_sampling.png')

    plot_custom('Accuracy', 'Latency-Energy Product', 'Accuracy (%)', 'Latency × Energy (kWh·h)',
                'Accuracy vs Latency-Energy Product', '4_accuracy_vs_latency_energy_product.png')

    plot_custom('Latency (h)', 'Accuracy/Energy', 'Latency (h)', 'Accuracy x Energy',
                'Latency vs Accuracy-Energy Product', '5_latency_vs_acc_energy_product.png')

    plot_custom('Total Energy (kWh)', 'Accuracy/Latency', 'Total Energy (kWh)', 'Accuracy x Latency',
                'Energy vs Accuracy-Latency Product', '6_energy_vs_acc_latency_product.png')

    # Log plots
    plot_custom('Latency (h)', 'Accuracy', 'Latency (h)', 'Accuracy (%)',
                'Accuracy vs Latency', '1_accuracy_vs_latency.png', logx=True)

    plot_custom('Total Energy (kWh)', 'Accuracy', 'Total Energy (kWh)', 'Accuracy (%)',
                'Accuracy vs Energy', '2_accuracy_vs_energy.png', logx=True)
    
    plot_custom('Sampling', 'Accuracy', 'Sampling (n)', 'Accuracy (%)',
                'Accuracy vs Sampling (n)', '3_accuracy_vs_sampling.png', logx=True, log_base=2)

    plot_custom('Accuracy', 'Latency-Energy Product', 'Accuracy (%)', 'Latency × Energy (kWh·h)',
                'Accuracy vs Latency-Energy Product', '4_accuracy_vs_latency_energy_product.png', logy=True)

    plot_custom('Latency (h)', 'Accuracy/Energy', 'Latency (h)', 'Accuracy x Energy',
                'Latency vs Accuracy-Energy Product', '5_latency_vs_acc_energy_product.png', logx=True, logy=True)

    plot_custom('Total Energy (kWh)', 'Accuracy/Latency', 'Total Energy (kWh)', 'Accuracy x Latency',
                'Energy vs Accuracy-Latency Product', '6_energy_vs_acc_latency_product.png', logx=True, logy=True)

    # Render legends
    render_legends()

    print(f"✅ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
