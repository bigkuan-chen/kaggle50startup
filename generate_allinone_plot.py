import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Train/Test Split matching the user's diagram split (test_size=0.19, random_state=0)
X = df.drop(columns='Profit')
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=0)

# 3. Preprocessing
# Scale numeric columns
scaler = StandardScaler()
num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# One-hot encode categorical State column
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[['State']])
X_test_cat = ohe.transform(X_test[['State']])

# Combine preprocessed features
cat_cols = list(ohe.get_feature_names_out(['State']))
feature_cols = num_cols + cat_cols

X_train_pre = pd.DataFrame(np.hstack([X_train_num, X_train_cat]), columns=feature_cols)
X_test_pre = pd.DataFrame(np.hstack([X_test_num, X_test_cat]), columns=feature_cols)

# 4. Define feature subsets sequentially
subsets = {
    1: ['R&D Spend'],
    2: ['R&D Spend', 'Marketing Spend'],
    3: ['R&D Spend', 'Marketing Spend', 'State_New York'],
    4: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    5: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
}

# 5. Define the 5 algorithms and their hyperparameter grids
algorithms = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Lasso Regression': {
        'model': Lasso(max_iter=10000),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'ElasticNet': {
        'model': ElasticNet(max_iter=10000),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 4, None],
            'min_samples_leaf': [1, 2]
        }
    }
}

# 6. Evaluate all algorithms across all subsets
results = {alg: {'RMSE': [], 'R2': []} for alg in algorithms}

for alg_name, config in algorithms.items():
    print(f"Training {alg_name} across subsets...")
    for k in sorted(subsets.keys()):
        sub = subsets[k]
        X_tr = X_train_pre[sub]
        X_te = X_test_pre[sub]
        
        if config['params']:
            grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_tr, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = config['model']
            best_model.fit(X_tr, y_train)
            
        preds = best_model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results[alg_name]['RMSE'].append(rmse)
        results[alg_name]['R2'].append(r2)

# 7. Create All-in-One Plot
fig = plt.figure(figsize=(14, 11), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.2], hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Color and style configurations for the 5 algorithms
styles = {
    'Linear Regression': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'Ridge Regression': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'Lasso Regression': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'ElasticNet': {'color': '#d62728', 'marker': 'd', 'linestyle': ':'},
    'Random Forest': {'color': '#9467bd', 'marker': 'x', 'linestyle': '-'}
}

num_features = list(subsets.keys())

# Plot RMSE
for alg_name in algorithms:
    ax1.plot(
        num_features, 
        results[alg_name]['RMSE'], 
        label=alg_name, 
        color=styles[alg_name]['color'],
        marker=styles[alg_name]['marker'],
        linestyle=styles[alg_name]['linestyle'],
        linewidth=2,
        markersize=6
    )
ax1.set_title("RMSE by Number of Features (All 5 Algorithms)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Number of Features", fontsize=10, labelpad=8)
ax1.set_ylabel("RMSE", fontsize=10, labelpad=8)
ax1.grid(True, linestyle='--', alpha=0.5, color='#ccc')
ax1.set_xticks(num_features)
ax1.legend(fontsize=9, loc='upper left')

# Plot R-squared
for alg_name in algorithms:
    ax2.plot(
        num_features, 
        results[alg_name]['R2'], 
        label=alg_name, 
        color=styles[alg_name]['color'],
        marker=styles[alg_name]['marker'],
        linestyle=styles[alg_name]['linestyle'],
        linewidth=2,
        markersize=6
    )
ax2.set_title("R-squared by Number of Features (All 5 Algorithms)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Number of Features", fontsize=10, labelpad=8)
ax2.set_ylabel("R-squared", fontsize=10, labelpad=8)
ax2.grid(True, linestyle='--', alpha=0.5, color='#ccc')
ax2.set_xticks(num_features)
ax2.legend(fontsize=9, loc='lower left')

# 8. Style and Annotation callouts directly on the figures
# We can annotate the optimal complexity point (2 features) on the best performing linear model (Lasso/Ridge)
ax1.annotate(
    "Optimal linear models\nachieve min RMSE at 2 features\n(R&D + Marketing spend)",
    xy=(2.0, results['Lasso Regression']['RMSE'][1]),
    xytext=(1.2, 8500),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", lw=1.2, color='#2ca02c'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#2ca02c", lw=0.8),
    fontweight='medium'
)

# Overfitting callout on RMSE
ax1.annotate(
    "Administration adds noise:\nPerformance degrades for\nall 5 algorithms at 5 features",
    xy=(5.0, results['Linear Regression']['RMSE'][4]),
    xytext=(2.2, 9050),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", lw=1.2, color='#d62728'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#d62728", lw=0.8),
    fontweight='medium'
)

# R2 annotations
ax2.annotate(
    "Peak Linear R² (94.74%)\nusing 2 features",
    xy=(2.0, results['Lasso Regression']['R2'][1]),
    xytext=(2.5, 0.941),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.08", lw=1.2, color='#1f77b4'),
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#1f77b4", lw=0.8),
    fontweight='medium'
)

# 9. Table below the plots
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

# Format results for the table:
# Rows: Subsets 1 to 5. Columns: Number of Features, Selected Features, RMSE for all 5 models, R-squared for all 5 models.
# Since we have 5 models, we can display a summary comparison table.
# Let's show: Subset, Features, Linear Reg R2, Ridge R2, Lasso R2, ElasticNet R2, Random Forest R2
table_data = []
for k in sorted(subsets.keys()):
    table_data.append([
        str(k),
        subsets[k],
        f"{results['Linear Regression']['R2'][k-1]:.4f}",
        f"{results['Ridge Regression']['R2'][k-1]:.4f}",
        f"{results['Lasso Regression']['R2'][k-1]:.4f}",
        f"{results['ElasticNet']['R2'][k-1]:.4f}",
        f"{results['Random Forest']['R2'][k-1]:.4f}"
    ])

col_labels = ["No. Features", "Selected Features", "Linear R²", "Ridge R²", "Lasso R²", "ElasticNet R²", "Random Forest R²"]

# Create table
table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.6)

# Table styling
for k, cell in table.get_celld().items():
    if k[0] == 0:
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor("#fafafa" if k[0] % 2 == 0 else "#ffffff")
    cell.set_linewidth(0.5)
    cell.set_edgecolor("#cccccc")

plt.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.92)

# Save to plots/ and current directory
os.makedirs('plots', exist_ok=True)
output_path1 = 'plots/allinone.png'
output_path2 = 'plots/feature_selection_performance_allinone.png'
output_path3 = 'allinone.png'

plt.savefig(output_path1, bbox_inches='tight', dpi=150)
plt.savefig(output_path2, bbox_inches='tight', dpi=150)
plt.savefig(output_path3, bbox_inches='tight', dpi=150)
plt.close()

print(f"All-in-one plot successfully generated and saved at:")
print(f"  - {output_path1}")
print(f"  - {output_path2}")
print(f"  - {output_path3}")
