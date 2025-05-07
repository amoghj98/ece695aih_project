import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Check command line args
if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <csv_file>")
    sys.exit(1)

csv_path = sys.argv[1]
csv_name = os.path.splitext(os.path.basename(csv_path))[0]
output_dir = os.path.join("plots", csv_name)
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Drop rows with missing values needed for plotting
df_clean = df.dropna(subset=['Latency (h)', 'Total Energy (kWh)', 'Accuracy']).copy()

# Compute derived metrics
df_clean['Latency-Energy Product'] = df_clean['Latency (h)'] * df_clean['Total Energy (kWh)']
df_clean['Accuracy/Energy'] = df_clean['Accuracy'] * df_clean['Total Energy (kWh)']
df_clean['Accuracy/Latency'] = df_clean['Accuracy'] * df_clean['Latency (h)']

# Color and marker maps
color_map = {'best of n': 'blue', 'beam search': 'green', 'dvts': 'red'}
marker_map = {4: '^', 16: 's', 64: 'D', 256: 'o'}

# Plot helper function
def plot_custom(x_col, y_col, xlabel, ylabel, title, filename):
    plt.figure(figsize=(9, 6))

    # Group by Model, Algo, and Hardware
    grouped = df_clean.groupby(['Model', 'Algo', 'Hardware'])

    for (model, algo, hw), group in grouped:
        group_sorted = group.sort_values('Sampling')
        color = color_map.get(algo, 'black')
        linestyle = ':'  # dotted line
        label_base = f"{model} ({hw})"

        for i, (_, row) in enumerate(group_sorted.iterrows()):
            sampling = row['Sampling']
            marker = marker_map.get(sampling, 'x')
            # Only show label once for the group (on first point)
            label = f"{model}, {algo}, {hw}" if i == 0 else None
            plt.plot(row[x_col], row[y_col], marker=marker, color=color,
                     linestyle='None', label=label)

        # Connect points with a dotted line for the curve
        plt.plot(group_sorted[x_col], group_sorted[y_col], linestyle=linestyle,
                 color=color, alpha=0.7)

    # De-duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize='x-small', frameon=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_custom('Latency (h)', 'Accuracy', 'Latency (h)', 'Accuracy (%)',
            '1. Accuracy vs Latency', '1_accuracy_vs_latency.png')

plot_custom('Total Energy (kWh)', 'Accuracy', 'Total Energy (kWh)', 'Accuracy (%)',
            '2. Accuracy vs Energy', '2_accuracy_vs_energy.png')

plot_custom('Accuracy', 'Latency-Energy Product', 'Accuracy (%)', 'Latency × Energy (kWh·h)',
            '3. Accuracy vs Latency-Energy Product', '3_accuracy_vs_latency_energy_product.png')

plot_custom('Latency (h)', 'Accuracy/Energy', 'Latency (h)', 'Accuracy x Energy',
            '4. Latency vs Accuracy-Energy Product', '4_latency_vs_acc_energy_product.png')

plot_custom('Total Energy (kWh)', 'Accuracy/Latency', 'Total Energy (kWh)', 'Accuracy x Latency',
            '5. Energy vs Accuracy-Latency Product', '5_energy_vs_acc_latency_product.png')

# Generate log-log plots
def plot_custom_loglog(x_col, y_col, xlabel, ylabel, title, filename):
    plt.figure(figsize=(9, 6))

    # Group by Model, Algo, and Hardware
    grouped = df_clean.groupby(['Model', 'Algo', 'Hardware'])

    for (model, algo, hw), group in grouped:
        group_sorted = group.sort_values('Sampling')
        color = color_map.get(algo, 'black')
        linestyle = ':'  # dotted line
        label_base = f"{model} ({hw})"

        for i, (_, row) in enumerate(group_sorted.iterrows()):
            sampling = row['Sampling']
            marker = marker_map.get(sampling, 'x')
            label = f"{model}, {algo}, {hw}" if i == 0 else None
            plt.plot(row[x_col], row[y_col], marker=marker, color=color,
                     linestyle='None', label=label)

        # Dotted line joining points
        plt.plot(group_sorted[x_col], group_sorted[y_col], linestyle=linestyle,
                 color=color, alpha=0.7)

    # Log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize='x-small', frameon=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 6. Latency vs Accuracy × Energy (log-log)
plot_custom_loglog('Latency (h)', 'Accuracy/Energy',
                   'Latency (h)', 'Accuracy × Energy',
                   '6. Latency vs Accuracy-Energy Product (Log-Log)',
                   '6_latency_vs_acc_energy_product_loglog.png')

# 7. Energy vs Accuracy × Latency (log-log)
plot_custom_loglog('Total Energy (kWh)', 'Accuracy/Latency',
                   'Total Energy (kWh)', 'Accuracy × Latency',
                   '7. Energy vs Accuracy-Latency Product (Log-Log)',
                   '7_energy_vs_acc_latency_product_loglog.png')

print(f"✅ All plots saved to: {output_dir}")
