import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from user's figure
num_features = [1, 2, 3, 4, 5]
features_list = [
    "[R&D Spend]",
    "[R&D Spend, Marketing Spend]",
    "[R&D Spend, Marketing Spend, New York]",
    "[R&D Spend, Marketing Spend, New York, Florida]",
    "[R&D Spend, Marketing Spend, New York, Florida, Administration]"
]
rmse_values = [8274.868018, 8198.797191, 8309.059683, 8409.916714, 9137.990153]
r2_values = [0.946459, 0.947439, 0.946015, 0.944697, 0.934707]

# Create figure
fig = plt.figure(figsize=(13, 9.5), dpi=150)

# Set up grid of 2 rows, where row 1 is for plots (left/right) and row 2 is for the table
gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1], hspace=0.3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Left plot: RMSE
ax1.plot(num_features, rmse_values, marker='o', color='#1f77b4', linewidth=2, markersize=6)
ax1.set_title("RMSE by Number of Features", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Number of Features", fontsize=10, labelpad=8)
ax1.set_ylabel("RMSE", fontsize=10, labelpad=8)
ax1.grid(True, linestyle='--', alpha=0.5, color='#ccc')
ax1.set_xticks(np.arange(1, 5.5, 0.5))
ax1.set_ylim(8100, 9300)

# Right plot: R-squared
ax2.plot(num_features, r2_values, marker='o', color='#1f77b4', linewidth=2, markersize=6)
ax2.set_title("R-squared by Number of Features", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Number of Features", fontsize=10, labelpad=8)
ax2.set_ylabel("R-squared", fontsize=10, labelpad=8)
ax2.grid(True, linestyle='--', alpha=0.5, color='#ccc')
ax2.set_xticks(np.arange(1, 5.5, 0.5))
ax2.set_ylim(0.932, 0.950)

# Add annotations to RMSE
# Optimal point (2 features)
ax1.annotate(
    "Optimal Complexity (2 features):\nMin RMSE ($8,198.80)\nFeatures: R&D + Marketing",
    xy=(2.0, 8198.797191),
    xytext=(1.2, 8550),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", lw=1.2, color='#2c3e50'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#eef2f3", ec="#95a5a6", lw=0.8),
    fontweight='medium'
)

# Overfitting / Administration point (5 features)
ax1.annotate(
    "Noise Overhead (5 features):\nAdding Administration spend\nincreases RMSE to $9,137.99 (+11.4%)",
    xy=(5.0, 9137.990153),
    xytext=(2.2, 8950),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.12", lw=1.2, color='#c0392b'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e59866", lw=0.8),
    fontweight='medium'
)

# Add annotations to R-squared
# 1 feature (R&D spend)
ax2.annotate(
    "R&D Spend Alone:\nExplains 94.65% variance!",
    xy=(1.0, 0.946459),
    xytext=(1.4, 0.938),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", lw=1.2, color='#27ae60'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#73c6b6", lw=0.8),
    fontweight='medium'
)

# Peak (2 features)
ax2.annotate(
    "Peak Performance:\nMax R²: 94.74%",
    xy=(2.0, 0.947439),
    xytext=(2.8, 0.946),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.08", lw=1.2, color='#1f77b4'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#5dade2", lw=0.8),
    fontweight='medium'
)

# Under plots: Add table
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

# Compile table data
table_data = []
for i in range(len(num_features)):
    table_data.append([
        str(num_features[i]),
        features_list[i],
        f"{rmse_values[i]:.6f}",
        f"{r2_values[i]:.6f}"
    ])

col_labels = ["Number of Features", "Selected Features", "RMSE", "R-squared"]

# Create table
table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Set colors matching user's visual style
for k, cell in table.get_celld().items():
    # Header cells
    if k[0] == 0:
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(weight='bold')
    else:
        # Alternating cell colors
        cell.set_facecolor("#fafafa" if k[0] % 2 == 0 else "#ffffff")
    cell.set_linewidth(0.5)
    cell.set_edgecolor("#cccccc")

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
plt.savefig('plots/feature_selection_annotated.png', bbox_inches='tight', dpi=150)
print("Annotated feature selection summary generated successfully at 'plots/feature_selection_annotated.png'")
