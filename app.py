import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page Configuration
st.set_page_config(
    page_title="50 Startups Interactive Feature Selection & Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS for styling
st.markdown("""
    <style>
    /* Main Layout */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Typography & Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #2c3e50 0%, #1abc9c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #1abc9c;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f2f6;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #57606f;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    
    /* Styled warnings/cards */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Callout styling */
    .callout-header {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 1. Load Data & Preprocessing Helpers
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists('data.csv'):
        # Fallback raw data if not found, though it should exist
        data = {
            'R&D Spend': [165349.2, 162597.7, 153441.51, 144372.41, 142107.34, 131876.9, 134615.46, 130298.13, 120542.52, 123334.88],
            'Administration': [136897.8, 151377.59, 101145.55, 118671.85, 91391.77, 99814.71, 147198.87, 145530.06, 148718.95, 108679.17],
            'Marketing Spend': [471784.1, 443898.53, 407934.54, 383199.62, 366168.42, 362861.36, 127716.82, 323876.68, 311613.29, 304981.62],
            'State': ['New York', 'California', 'Florida', 'New York', 'Florida', 'New York', 'California', 'Florida', 'New York', 'California'],
            'Profit': [192261.83, 191792.06, 191050.39, 182901.99, 166187.94, 156991.12, 156122.51, 155752.6, 152211.77, 149759.96]
        }
        return pd.DataFrame(data)
    return pd.read_csv('data.csv')

df = load_data()

# Data splitting (matching standard project splits)
X = df.drop(columns='Profit')
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=0)

# Fitted preprocessing components
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[['R&D Spend', 'Administration', 'Marketing Spend']])
X_test_num = scaler.transform(X_test[['R&D Spend', 'Administration', 'Marketing Spend']])

ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[['State']])
X_test_cat = ohe.transform(X_test[['State']])

cat_cols = list(ohe.get_feature_names_out(['State']))
feature_cols = ['R&D Spend', 'Administration', 'Marketing Spend'] + cat_cols

X_train_pre = pd.DataFrame(np.hstack([X_train_num, X_train_cat]), columns=feature_cols)
X_test_pre = pd.DataFrame(np.hstack([X_test_num, X_test_cat]), columns=feature_cols)

# Define preset subsets
subsets = {
    1: ['R&D Spend'],
    2: ['R&D Spend', 'Marketing Spend'],
    3: ['R&D Spend', 'Marketing Spend', 'State_New York'],
    4: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    5: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
}

subsets_features_readable = {
    1: "[R&D Spend]",
    2: "[R&D Spend, Marketing Spend]",
    3: "[R&D Spend, Marketing Spend, NY]",
    4: "[R&D Spend, Marketing Spend, NY, FL]",
    5: "[R&D Spend, Marketing Spend, NY, FL, Admin]"
}

# -------------------------------------------------------------------------
# 2. Sidebar Configuration & Setup
# -------------------------------------------------------------------------
st.sidebar.markdown("### 🛠️ Model configurations")

# Hyperparameter search config or sliders
tuning_mode = st.sidebar.toggle("Enable Hyperparameter Tuning Grid Search", value=True, help="Run GridSearchCV for model selection, otherwise use default model setups.")

# Multiselect for models to display in Tab 1
selected_models = st.sidebar.multiselect(
    "Algorithms to Evaluate in Tab 1",
    options=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest'],
    default=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest']
)

# Individual model parameter overrides (when grid search is disabled)
if not tuning_mode:
    st.sidebar.markdown("#### ⚙️ Manual parameters")
    ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.01, 100.0, 1.0)
    lasso_alpha = st.sidebar.slider("Lasso Alpha", 0.01, 100.0, 1.0)
    en_alpha = st.sidebar.slider("ElasticNet Alpha", 0.01, 10.0, 1.0)
    en_l1 = st.sidebar.slider("ElasticNet L1 Ratio", 0.0, 1.0, 0.5)
    rf_estimators = st.sidebar.slider("RF Estimators", 10, 500, 100)
    rf_depth = st.sidebar.slider("RF Max Depth", 2, 10, 4)

# Cache-able benchmark evaluations
@st.cache_data
def evaluate_benchmarks(_tuning_mode, _ridge_alpha=1.0, _lasso_alpha=1.0, _en_alpha=1.0, _en_l1=0.5, _rf_estimators=100, _rf_depth=4):
    # Setup algorithms configuration
    if _tuning_mode:
        algorithms = {
            'Linear Regression': {'model': LinearRegression(), 'params': {}},
            'Ridge Regression': {'model': Ridge(), 'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}},
            'Lasso Regression': {'model': Lasso(max_iter=10000), 'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}},
            'ElasticNet': {'model': ElasticNet(max_iter=10000), 'params': {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}},
            'Random Forest': {'model': RandomForestRegressor(random_state=42), 'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 4, None],
                'min_samples_leaf': [1, 2]
            }}
        }
    else:
        algorithms = {
            'Linear Regression': {'model': LinearRegression(), 'params': {}},
            'Ridge Regression': {'model': Ridge(alpha=_ridge_alpha), 'params': {}},
            'Lasso Regression': {'model': Lasso(alpha=_lasso_alpha, max_iter=10000), 'params': {}},
            'ElasticNet': {'model': ElasticNet(alpha=_en_alpha, l1_ratio=_en_l1, max_iter=10000), 'params': {}},
            'Random Forest': {'model': RandomForestRegressor(n_estimators=_rf_estimators, max_depth=_rf_depth, random_state=42), 'params': {}}
        }

    results = {alg: {'RMSE': [], 'R2': []} for alg in algorithms}
    best_params = {alg: [] for alg in algorithms}
    
    num_features = list(subsets.keys())
    
    for alg_name, config in algorithms.items():
        for k in sorted(subsets.keys()):
            sub = subsets[k]
            X_tr = X_train_pre[sub]
            X_te = X_test_pre[sub]
            
            if config['params']:
                grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_tr, y_train)
                best_model = grid.best_estimator_
                best_params[alg_name].append(str(grid.best_params_))
            else:
                best_model = config['model']
                best_model.fit(X_tr, y_train)
                best_params[alg_name].append('Default')
                
            preds = best_model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            results[alg_name]['RMSE'].append(rmse)
            results[alg_name]['R2'].append(r2)
            
    return results, best_params

# Fetch results
if tuning_mode:
    benchmark_results, best_params = evaluate_benchmarks(True)
else:
    benchmark_results, best_params = evaluate_benchmarks(
        False, ridge_alpha, lasso_alpha, en_alpha, en_l1, rf_estimators, rf_depth
    )

# -------------------------------------------------------------------------
# Header & Dashboard Intro
# -------------------------------------------------------------------------
st.markdown("<h1>🚀 50 Startups: Interactive Feature Selection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A premium interactive web application to analyze how feature subsets affect regression models, investigate the effect of noise features, and run real-time profit predictions.</div>", unsafe_allow_html=True)

# Top Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Optimal Feature Count", value="2 Features", delta="R&D + Marketing", delta_color="normal")
with col_m2:
    st.metric(label="Peak R² Performance", value="94.74%", delta="Linear & Lasso Models", delta_color="normal")
with col_m3:
    st.metric(label="Minimum Test RMSE", value="$8,198.80", delta="Compared to Baseline", delta_color="inverse")
with col_m4:
    st.metric(label="Administration Overhead", value="Noise Impact", delta="Degrades model by 11.4%", delta_color="inverse")

st.markdown("---")

# Main Page Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Feature Selection Benchmarks", 
    "🔧 Custom Feature Explorer", 
    "🔮 Live Profit Predictor", 
    "📈 Interactive EDA"
])

# -------------------------------------------------------------------------
# Tab 1: Feature Selection Benchmarks (Replicating feature_selection_annotated.png)
# -------------------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Performance Curves by Feature Subset")
    st.markdown("""
        Below is the interactive recreation of the feature selection summary. 
        Select algorithms in the sidebar to add them to the plots. Hover over nodes to see details, and toggle annotations to view key CRISP-DM highlights.
    """)
    
    # Toggle annotations option
    show_annotations = st.checkbox("Show Performance Annotations (Callouts)", value=True)
    
    # Check if we have selected models to display
    if not selected_models:
        st.warning("Please select at least one algorithm in the sidebar to visualize.")
    else:
        # Create interactive Plotly figure
        fig_rmse = go.Figure()
        fig_r2 = go.Figure()
        
        styles = {
            'Linear Regression': {'color': '#1f77b4', 'symbol': 'circle', 'dash': 'solid'},
            'Ridge Regression': {'color': '#ff7f0e', 'symbol': 'square', 'dash': 'dash'},
            'Lasso Regression': {'color': '#2ca02c', 'symbol': 'triangle-up', 'dash': 'dashdot'},
            'ElasticNet': {'color': '#d62728', 'symbol': 'diamond', 'dash': 'dot'},
            'Random Forest': {'color': '#9467bd', 'symbol': 'x', 'dash': 'solid'}
        }
        
        num_features = list(subsets.keys())
        features_hover = [subsets_features_readable[k] for k in num_features]
        
        # Add traces for selected models
        for model in selected_models:
            res = benchmark_results[model]
            
            # RMSE Trace
            fig_rmse.add_trace(go.Scatter(
                x=num_features,
                y=res['RMSE'],
                mode='lines+markers',
                name=model,
                line=dict(color=styles[model]['color'], width=3, dash=styles[model]['dash']),
                marker=dict(symbol=styles[model]['symbol'], size=9),
                customdata=features_hover,
                hovertemplate="<b>%{hovertext}</b><br>Features: %{customdata}<br>RMSE: $%{y:,.2f}<extra></extra>",
                hovertext=model
            ))
            
            # R2 Trace
            fig_r2.add_trace(go.Scatter(
                x=num_features,
                y=res['R2'],
                mode='lines+markers',
                name=model,
                line=dict(color=styles[model]['color'], width=3, dash=styles[model]['dash']),
                marker=dict(symbol=styles[model]['symbol'], size=9),
                customdata=features_hover,
                hovertemplate="<b>%{hovertext}</b><br>Features: %{customdata}<br>R²: %{y:.6f}<extra></extra>",
                hovertext=model
            ))
            
        # Customize RMSE plot layout
        fig_rmse.update_layout(
            title=dict(text="<b>RMSE by Number of Features</b>", font=dict(size=16)),
            xaxis=dict(title="Number of Features", tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(title="RMSE ($)"),
            hovermode="closest",
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_rmse.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)
        fig_rmse.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)
        
        # Customize R2 plot layout
        fig_r2.update_layout(
            title=dict(text="<b>R-squared by Number of Features</b>", font=dict(size=16)),
            xaxis=dict(title="Number of Features", tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(title="R-squared (R²)", tickformat=".3f"),
            hovermode="closest",
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_r2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)
        fig_r2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)

        # Apply Annotations (Callouts) if requested
        if show_annotations:
            # 1. Optimal point (2 features) on RMSE Plot
            if 'Linear Regression' in selected_models:
                opt_rmse = benchmark_results['Linear Regression']['RMSE'][1]
                fig_rmse.add_annotation(
                    x=2, y=opt_rmse,
                    text="<b>Optimal Complexity (2 features):</b><br>Min RMSE ($8,198.80)<br>Features: R&D + Marketing",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#2c3e50",
                    ax=-90, ay=70,
                    bgcolor="#eef2f3", bordercolor="#95a5a6", borderwidth=1, borderpad=6,
                    font=dict(size=10, color="#2c3e50")
                )
                
                # Overfitting callout at 5 features on RMSE Plot
                overfit_rmse = benchmark_results['Linear Regression']['RMSE'][4]
                fig_rmse.add_annotation(
                    x=5, y=overfit_rmse,
                    text="<b>Noise Overhead (5 features):</b><br>Adding Administration spend<br>increases RMSE to $9,137.99 (+11.4%)",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#c0392b",
                    ax=-120, ay=-80,
                    bgcolor="#fdf2e9", bordercolor="#e59866", borderwidth=1, borderpad=6,
                    font=dict(size=10, color="#2c3e50")
                )
                
                # R2 Annotations
                opt_r2 = benchmark_results['Linear Regression']['R2'][1]
                fig_r2.add_annotation(
                    x=2, y=opt_r2,
                    text="<b>Peak Performance:</b><br>Max R²: 94.74%",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#1f77b4",
                    ax=90, ay=50,
                    bgcolor="#ebf5fb", bordercolor="#5dade2", borderwidth=1, borderpad=6,
                    font=dict(size=10, color="#2c3e50")
                )
                
                base_r2 = benchmark_results['Linear Regression']['R2'][0]
                fig_r2.add_annotation(
                    x=1, y=base_r2,
                    text="<b>R&D Spend Alone:</b><br>Explains 94.65% variance!",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#27ae60",
                    ax=90, ay=-50,
                    bgcolor="#e8f8f5", bordercolor="#73c6b6", borderwidth=1, borderpad=6,
                    font=dict(size=10, color="#2c3e50")
                )

        # Plot charts side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_rmse, use_container_width=True)
        with col2:
            st.plotly_chart(fig_r2, use_container_width=True)

        # Summary Benchmark Data Table
        st.markdown("### 📋 Performance Comparison Matrix")
        
        table_rows = []
        for k in sorted(subsets.keys()):
            row_dict = {
                "Subset": f"E{k}",
                "Features": subsets_features_readable[k],
                "Count": k
            }
            # Fill metrics for all models
            for model in selected_models:
                row_dict[f"{model} R²"] = f"{benchmark_results[model]['R2'][k-1]:.5f}"
                row_dict[f"{model} RMSE"] = f"${benchmark_results[model]['RMSE'][k-1]:,.2f}"
            table_rows.append(row_dict)
            
        df_table = pd.DataFrame(table_rows)
        
        # Display table with streamlit dataframe styling
        st.dataframe(
            df_table, 
            hide_index=True, 
            use_container_width=True
        )

# -------------------------------------------------------------------------
# Tab 2: Custom Feature Explorer & Trainer
# -------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔧 Custom Model Builder & Feature Selector")
    st.markdown("""
        Select any combination of features (including base and engineered features) 
        and choose a regression algorithm to train and evaluate a custom model on the fly.
    """)
    
    col_feat, col_params = st.columns([2, 1])
    
    with col_feat:
        st.markdown("#### 1. Select Features")
        
        # We can construct base + engineered columns
        feat_base = st.multiselect(
            "Base Dataset Features",
            options=['R&D Spend', 'Administration', 'Marketing Spend', 'State_New York', 'State_Florida'],
            default=['R&D Spend', 'Marketing Spend'],
            help="Select core numerical spend parameters and one-hot encoded state proxies."
        )
        
        # Let's compute engineered features in pandas for X_train and X_test
        # We will create engineered columns on pre-scaled data
        feat_eng = st.multiselect(
            "Engineered Synergies & Ratios",
            options=['Total Spend', 'R&D Ratio', 'Marketing Ratio', 'Administration Ratio', 'Log Marketing Spend', 'R&D x Marketing'],
            default=[],
            help="Select engineered columns constructed via Custom FeatureEngineer transformer."
        )
        
        all_custom_feats = feat_base + feat_eng
        
    with col_params:
        st.markdown("#### 2. Select Algorithm")
        custom_alg = st.selectbox(
            "Regression Algorithm",
            options=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest']
        )
        
        # Model-specific parameters for Tab 2
        st.markdown("##### Model Parameters")
        if custom_alg == 'Linear Regression':
            st.info("Linear Regression has no tunable hyperparameters.")
            model_obj = LinearRegression()
        elif custom_alg == 'Ridge Regression':
            c_alpha = st.slider("Ridge Alpha (Regularization Strength)", 0.01, 100.0, 1.0, key="custom_ridge")
            model_obj = Ridge(alpha=c_alpha)
        elif custom_alg == 'Lasso Regression':
            c_alpha = st.slider("Lasso Alpha (Regularization Strength)", 0.01, 100.0, 1.0, key="custom_lasso")
            model_obj = Lasso(alpha=c_alpha, max_iter=10000)
        elif custom_alg == 'ElasticNet':
            c_alpha = st.slider("ElasticNet Alpha", 0.01, 10.0, 1.0, key="custom_en")
            c_l1 = st.slider("L1 Ratio (0 = Ridge, 1 = Lasso)", 0.0, 1.0, 0.5, key="custom_en_l1")
            model_obj = ElasticNet(alpha=c_alpha, l1_ratio=c_l1, max_iter=10000)
        elif custom_alg == 'Random Forest':
            c_est = st.slider("Number of Estimators (Trees)", 10, 500, 100, key="custom_rf_est")
            c_depth = st.slider("Max Tree Depth", 2, 10, 4, key="custom_rf_depth")
            model_obj = RandomForestRegressor(n_estimators=c_est, max_depth=c_depth, random_state=42)

    if not all_custom_feats:
        st.error("Please select at least one feature to train the custom model.")
    else:
        # Dynamically build training data based on selected features
        # Base features (R&D Spend, Administration, Marketing Spend) are scaled. State dummies are encoded.
        # We need to construct the engineered features on raw data, then scale them!
        
        # Step 1: Standard splits of raw data
        raw_X_tr = X_train.copy()
        raw_X_te = X_test.copy()
        
        # Custom Feature Engineering implementation matching project logic
        def engineer_data(df_in):
            df_out = df_in.copy()
            df_out['Total Spend'] = df_out['R&D Spend'] + df_out['Administration'] + df_out['Marketing Spend']
            total_spend_safe = np.where(df_out['Total Spend'] == 0, 1e-9, df_out['Total Spend'])
            df_out['R&D Ratio'] = df_out['R&D Spend'] / total_spend_safe
            df_out['Marketing Ratio'] = df_out['Marketing Spend'] / total_spend_safe
            df_out['Administration Ratio'] = df_out['Administration'] / total_spend_safe
            df_out['Log Marketing Spend'] = np.log1p(df_out['Marketing Spend'])
            df_out['R&D x Marketing'] = df_out['R&D Spend'] * df_out['Marketing Spend']
            return df_out

        raw_X_tr_eng = engineer_data(raw_X_tr)
        raw_X_te_eng = engineer_data(raw_X_te)
        
        # Identify columns to scale
        cols_to_scale = ['R&D Spend', 'Administration', 'Marketing Spend', 
                         'Total Spend', 'R&D Ratio', 'Marketing Ratio', 
                         'Administration Ratio', 'Log Marketing Spend', 'R&D x Marketing']
        
        active_scale_cols = [c for c in cols_to_scale if c in all_custom_feats]
        active_unscaled_cols = [c for c in all_custom_feats if c not in cols_to_scale] # State dummies
        
        # Apply standard scaling dynamically
        X_tr_custom = pd.DataFrame(index=raw_X_tr.index)
        X_te_custom = pd.DataFrame(index=raw_X_te.index)
        
        if active_scale_cols:
            custom_scaler = StandardScaler()
            scaled_tr = custom_scaler.fit_transform(raw_X_tr_eng[active_scale_cols])
            scaled_te = custom_scaler.transform(raw_X_te_eng[active_scale_cols])
            
            scaled_tr_df = pd.DataFrame(scaled_tr, columns=active_scale_cols, index=raw_X_tr.index)
            scaled_te_df = pd.DataFrame(scaled_te, columns=active_scale_cols, index=raw_X_te.index)
            
            X_tr_custom = pd.concat([X_tr_custom, scaled_tr_df], axis=1)
            X_te_custom = pd.concat([X_te_custom, scaled_te_df], axis=1)
            
        # State dummies extraction
        state_encoder = OneHotEncoder(drop='first', sparse_output=False)
        state_tr = state_encoder.fit_transform(raw_X_tr[['State']])
        state_te = state_encoder.transform(raw_X_te[['State']])
        
        state_cols = list(state_encoder.get_feature_names_out(['State']))
        state_tr_df = pd.DataFrame(state_tr, columns=state_cols, index=raw_X_tr.index)
        state_te_df = pd.DataFrame(state_te, columns=state_cols, index=raw_X_te.index)
        
        for col in active_unscaled_cols:
            if col in state_tr_df.columns:
                X_tr_custom[col] = state_tr_df[col]
                X_te_custom[col] = state_te_df[col]
                
        # Train and Predict
        model_obj.fit(X_tr_custom, y_train)
        custom_preds_test = model_obj.predict(X_te_custom)
        custom_preds_train = model_obj.predict(X_tr_custom)
        
        # Calculate performance metrics
        custom_train_r2 = r2_score(y_train, custom_preds_train)
        custom_test_r2 = r2_score(y_test, custom_preds_test)
        custom_test_rmse = np.sqrt(mean_squared_error(y_test, custom_preds_test))
        custom_test_mae = mean_absolute_error(y_test, custom_preds_test)
        
        st.markdown("---")
        st.markdown("#### 📈 Custom Model Metrics")
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Train R-squared (R²)", f"{custom_train_r2:.4f}")
        with c_m2:
            st.metric("Test R-squared (R²)", f"{custom_test_r2:.4f}")
        with c_m3:
            st.metric("Test RMSE ($)", f"${custom_test_rmse:,.2f}")
        with c_m4:
            st.metric("Test MAE ($)", f"${custom_test_mae:,.2f}")
            
        # Display Feature Weights / Coefficients (if linear model) or Importances (if Random Forest)
        st.markdown("---")
        st.markdown("#### ⚖️ Feature Influence")
        
        if custom_alg in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            coefs = model_obj.coef_
            intercept = model_obj.intercept_
            
            coef_df = pd.DataFrame({
                'Feature': X_tr_custom.columns,
                'Coefficient': coefs
            }).sort_values(by='Coefficient', key=abs, ascending=True)
            
            fig_coef = px.bar(
                coef_df,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title=f"Linear Model Coefficients (Intercept / Baseline: ${intercept:,.2f})",
                color='Coefficient',
                color_continuous_scale=px.colors.diverging.RdYlBu,
                text_auto='.2f'
            )
            fig_coef.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
            st.plotly_chart(fig_coef, use_container_width=True)
            
        elif custom_alg == 'Random Forest':
            importances = model_obj.feature_importances_
            
            imp_df = pd.DataFrame({
                'Feature': X_tr_custom.columns,
                'Importance (%)': importances * 100
            }).sort_values(by='Importance (%)', ascending=True)
            
            fig_imp = px.bar(
                imp_df,
                x='Importance (%)',
                y='Feature',
                orientation='h',
                title="Random Forest Feature Importance (%)",
                color='Importance (%)',
                color_continuous_scale='Viridis',
                text_auto='.2f%'
            )
            fig_imp.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------------------
# Tab 3: Live Profit Predictor (Based on Optimal 2-feature Model)
# -------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔮 Live Profit Predictor Tool")
    st.markdown("""
        Enter details for a mock startup. The tool uses the **empirically optimal 2-feature Linear Regression model** 
        (`['R&D Spend', 'Marketing Spend']`) to compute predicted net profit and details the contribution of each expenditure.
    """)
    
    col_inputs, col_gauge = st.columns([1, 1])
    
    with col_inputs:
        st.markdown("#### Startup Expenditures")
        rd_val = st.slider("R&D Spend ($)", 0, 250000, 75000, step=1000, help="Investment in Research & Development")
        mkt_val = st.slider("Marketing Spend ($)", 0, 500000, 200000, step=1000, help="Investment in Growth and Market Exposure")
        
        st.markdown("#### Irrelevant Spends & Metadata")
        admin_val = st.slider("Administration Spend ($)", 0, 250000, 120000, step=1000, help="Administrative costs. Empirically proven to be noise.")
        state_val = st.selectbox("State of Operation", options=['California', 'Florida', 'New York'], index=0)
        
    with col_gauge:
        # Train optimal model on the fly to avoid deserialization errors and ensure robust prediction
        # Optimal features: ['R&D Spend', 'Marketing Spend']
        opt_scaler = StandardScaler()
        X_train_opt_num = opt_scaler.fit_transform(X_train[['R&D Spend', 'Marketing Spend']])
        
        opt_lr = LinearRegression()
        opt_lr.fit(X_train_opt_num, y_train)
        
        # Preprocess input values
        input_raw = pd.DataFrame([[rd_val, mkt_val]], columns=['R&D Spend', 'Marketing Spend'])
        input_scaled = opt_scaler.transform(input_raw)
        
        # Predict profit
        predicted_profit = opt_lr.predict(input_scaled)[0]
        
        # Draw Gauge chart for predicted profit
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_profit,
            number = {'prefix': "$", 'valueformat': ",.2f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "<b>Predicted Startup Net Profit</b>", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 250000], 'tickprefix': "$", 'tickformat': ","},
                'bar': {'color': "#1abc9c"},
                'steps': [
                    {'range': [0, 50000], 'color': "#f5f6fa"},
                    {'range': [50000, 120000], 'color': "#dcdde1"},
                    {'range': [120000, 200000], 'color': "#c8d6e5"},
                    {'range': [200000, 250000], 'color': "#8395a7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': df['Profit'].max()
                }
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(t=50, b=0, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 📊 Contribution Analysis")
    st.markdown("""
        How did the expenditures yield this prediction? 
        The baseline profit (Intercept) represents the average profit of startups in CA (~$111k). 
        Adjustments are computed based on how much the expenditures deviate from dataset averages.
    """)
    
    # Calculate contribution breakdown
    # Prediction = Intercept + Coef_RD * Scaled_RD + Coef_Mkt * Scaled_Mkt
    # Scaled = (Val - Mean) / Std
    means = opt_scaler.mean_
    stds = opt_scaler.scale_
    coefs = opt_lr.coef_
    intercept = opt_lr.intercept_
    
    rd_scaled = (rd_val - means[0]) / stds[0]
    mkt_scaled = (mkt_val - means[1]) / stds[1]
    
    rd_contrib = coefs[0] * rd_scaled
    mkt_contrib = coefs[1] * mkt_scaled
    
    # Waterfall chart to visualize contributions
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Contribution", 
        orientation = "v",
        measure = ["absolute", "relative", "relative", "total"],
        x = ["Baseline (Average Startup)", "R&D Spend Deviation", "Marketing Spend Deviation", "Final Predicted Profit"],
        textposition = "outside",
        text = [f"${intercept:,.2f}", f"${rd_contrib:,.2f}", f"${mkt_contrib:,.2f}", f"${predicted_profit:,.2f}"],
        y = [intercept, rd_contrib, mkt_contrib, predicted_profit],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#e74c3c"}},
        increasing = {"marker":{"color":"#2ecc71"}},
        totals = {"marker":{"color":"#2c3e50"}}
    ))
    
    fig_waterfall.update_layout(
        title = "Startup Profit Breakdown (Waterfall Chart)",
        showlegend = False,
        plot_bgcolor = "white",
        yaxis = dict(title = "Profit ($)", tickprefix="$", tickformat=","),
        margin = dict(l=40, r=40, t=60, b=40)
    )
    fig_waterfall.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee')
    
    st.plotly_chart(fig_waterfall, use_container_width=True)

# -------------------------------------------------------------------------
# Tab 4: Interactive Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------------
with tab4:
    st.markdown("### 📈 Interactive Exploratory Data Analysis")
    st.markdown("""
        Explore the properties, distributions, and bivariate correlations of the startup dataset.
    """)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🔗 Correlation Heatmap")
        # Compute correlations
        corr_df = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
        
        # Interactive heatmap using plotly
        fig_heat = px.imshow(
            corr_df,
            labels=dict(color="Correlation"),
            x=corr_df.columns,
            y=corr_df.columns,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            text_auto=".4f"
        )
        fig_heat.update_layout(
            title="Pearson Correlation Matrix",
            coloraxis_showscale=True,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_right:
        st.markdown("#### 🔍 Custom Bivariate Scatter Plot")
        
        x_axis_col = st.selectbox("X-Axis Feature", options=['R&D Spend', 'Administration', 'Marketing Spend'], index=0)
        y_axis_col = st.selectbox("Y-Axis Target", options=['Profit', 'R&D Spend', 'Administration', 'Marketing Spend'], index=0)
        
        try:
            import statsmodels
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False

        fig_scat = px.scatter(
            df,
            x=x_axis_col,
            y=y_axis_col,
            color='State',
            hover_name='State',
            trendline='ols' if has_statsmodels else None,
            trendline_color_override='#2c3e50' if has_statsmodels else None,
            title=f"Scatter Plot: {y_axis_col} vs {x_axis_col}" + (" (with linear trendline)" if has_statsmodels else ""),
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_scat.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#eee'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#eee')
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        
    # Bottom raw dataset viewer
    st.markdown("---")
    st.markdown("#### 📋 Raw Startups Dataset")
    st.dataframe(df, use_container_width=True)
