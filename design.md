project:
  title: "Kaggle 50 Startups Profit Prediction"
  framework: "CRISP-DM"
  goal: "Predict startup Profit and explain how spending allocation affects profit."
  problem_type: "Supervised Learning - Regression"

dataset:
  source: "Kaggle 50 Startups"
  rows: 50
  target: "Profit"
  features:
    numeric:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
    categorical:
      - "State"

expert_considerations:
  dataset_limitations:
    - "Only 50 rows, so overfitting risk is high."
    - "This project should focus on interpretation, not only prediction score."
    - "Deep learning is not suitable for this dataset."
  business_questions:
    - "Which spending category affects Profit the most?"
    - "Is Administration spending useful or mostly noise?"
    - "Does State truly affect Profit or act as a proxy variable?"
    - "Does spending ratio explain Profit better than raw spending amount?"
  expected_patterns:
    - "R&D Spend is likely the strongest predictor."
    - "Marketing Spend may have moderate positive impact."
    - "Administration may have weak relationship with Profit."
    - "State should be interpreted carefully."

crisp_dm:

  1_business_understanding:
    objective: "Help startups understand how budget allocation relates to Profit."
    business_problem: "How should a startup allocate R&D, Marketing, and Administration spending to improve Profit?"
    ml_problem: "Regression model to predict Profit."
    success_criteria:
      technical:
        - "High cross-validation R2"
        - "Low MAE and RMSE"
        - "Stable model performance across folds"
      business:
        - "Clear explanation of important features"
        - "Actionable budget allocation insights"
        - "Avoid overclaiming causality"

  2_data_understanding:
    tasks:
      - name: "Load dataset"
        output: "DataFrame"
      - name: "Check shape"
        expected: "50 rows and 5 columns"
      - name: "Check columns"
        expected_columns:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
          - "State"
          - "Profit"
      - name: "Check data types"
        numeric_columns:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
          - "Profit"
        categorical_columns:
          - "State"
      - name: "Check missing values"
        action: "Report missing count per column"
      - name: "Check duplicates"
        action: "Report duplicate row count"
      - name: "Explore basic statistics"
        action: "Use describe() for numeric columns"

    eda:
      univariate:
        - "Histogram of Profit"
        - "Histogram of R&D Spend"
        - "Histogram of Administration"
        - "Histogram of Marketing Spend"
      bivariate:
        - "Scatter plot: R&D Spend vs Profit"
        - "Scatter plot: Marketing Spend vs Profit"
        - "Scatter plot: Administration vs Profit"
        - "Boxplot: Profit by State"
      correlation:
        method: "Pearson correlation"
        focus:
          - "Correlation between R&D Spend and Profit"
          - "Correlation between Marketing Spend and Profit"
          - "Correlation between Administration and Profit"

  3_data_preparation:
    preprocessing:
      missing_values:
        strategy: "Check first; if missing values exist, use median for numeric and mode for categorical."
      duplicates:
        strategy: "Remove duplicated rows if found."
      categorical_encoding:
        column: "State"
        method: "OneHotEncoder"
        drop: "first"
        reason: "Avoid dummy variable trap for linear models."
      numeric_scaling:
        method: "StandardScaler"
        apply_to:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
          - "Total Spend"
          - "R&D Ratio"
          - "Marketing Ratio"
          - "Administration Ratio"
        reason: "Useful for Ridge and Lasso regression."

    feature_engineering:
      - name: "Total Spend"
        formula: "R&D Spend + Administration + Marketing Spend"
        purpose: "Measure total investment scale."
      - name: "R&D Ratio"
        formula: "R&D Spend / Total Spend"
        purpose: "Measure budget share allocated to product and innovation."
      - name: "Marketing Ratio"
        formula: "Marketing Spend / Total Spend"
        purpose: "Measure budget share allocated to market growth."
      - name: "Administration Ratio"
        formula: "Administration / Total Spend"
        purpose: "Measure budget share allocated to internal operations."
      - name: "Log Marketing Spend"
        formula: "log1p(Marketing Spend)"
        purpose: "Test possible diminishing returns of marketing."
      - name: "R&D x Marketing"
        formula: "R&D Spend * Marketing Spend"
        purpose: "Test interaction between product investment and market exposure."

    train_test_split:
      test_size: 0.2
      random_state: 42
      note: "Because dataset is small, final judgment should rely more on cross-validation than one split."

  4_modeling:
    recommended_models:
      - name: "Linear Regression"
        role: "Baseline interpretable model"
        sklearn_class: "sklearn.linear_model.LinearRegression"
      - name: "Ridge Regression"
        role: "Regularized linear model to reduce overfitting"
        sklearn_class: "sklearn.linear_model.Ridge"
        hyperparameters:
          alpha: [0.01, 0.1, 1, 10, 100]
      - name: "Lasso Regression"
        role: "Feature selection and interpretation"
        sklearn_class: "sklearn.linear_model.Lasso"
        hyperparameters:
          alpha: [0.001, 0.01, 0.1, 1, 10]
      - name: "Random Forest Regressor"
        role: "Nonlinear comparison and feature importance"
        sklearn_class: "sklearn.ensemble.RandomForestRegressor"
        hyperparameters:
          n_estimators: [100, 300, 500]
          max_depth: [2, 3, 4, null]
          min_samples_leaf: [1, 2, 4]

    not_recommended:
      - name: "Neural Network"
        reason: "Dataset is too small."
      - name: "Deep Learning"
        reason: "High overfitting risk."
      - name: "Transformer"
        reason: "Not appropriate for a 50-row tabular regression problem."

  5_evaluation:
    metrics:
      - name: "MAE"
        sklearn_function: "mean_absolute_error"
        interpretation: "Average absolute prediction error."
      - name: "RMSE"
        sklearn_function: "mean_squared_error with squared=False"
        interpretation: "Penalizes large prediction errors."
      - name: "R2"
        sklearn_function: "r2_score"
        interpretation: "Explained variance."
      - name: "Cross Validation R2"
        sklearn_function: "cross_val_score"
        folds: 5
        interpretation: "More reliable than one train/test split for small data."

    model_selection_criteria:
      primary:
        - "Cross-validation R2"
        - "MAE"
        - "Model interpretability"
      secondary:
        - "RMSE"
        - "Feature importance stability"
      warning:
        - "Do not choose a complex model only because it performs slightly better on one train/test split."

  6_interpretation:
    analyses:
      - name: "Linear coefficients"
        purpose: "Understand direction and size of each feature effect."
      - name: "Ridge coefficients"
        purpose: "Check stable linear effects after regularization."
      - name: "Lasso selected features"
        purpose: "Identify features that may be removed."
      - name: "Random Forest feature importance"
        purpose: "Check nonlinear importance ranking."
      - name: "Compare model with and without Administration"
        purpose: "Test whether Administration is useful or noise."
      - name: "Compare raw spending features vs ratio features"
        purpose: "Test whether investment structure matters."

    expected_insights:
      - "R&D Spend should likely be the strongest predictor."
      - "Administration may have weak or unstable importance."
      - "Marketing Spend may be useful but weaker than R&D."
      - "State should not be over-interpreted as causal."
      - "Budget ratios may provide stronger business insight than raw spending alone."

  7_deployment:
    deliverables:
      - "Clean sklearn pipeline"
      - "EDA charts"
      - "Model comparison table"
      - "Feature importance table"
      - "Final business recommendation"
    final_output:
      prediction_function:
        input:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
          - "State"
        output:
          - "Predicted Profit"
      business_report:
        sections:
          - "Project objective"
          - "Dataset limitation"
          - "Model performance"
          - "Feature interpretation"
          - "Budget allocation recommendation"

sklearn_pipeline_design:
  modules:
    - "pandas"
    - "numpy"
    - "matplotlib"
    - "seaborn"
    - "sklearn.model_selection"
    - "sklearn.preprocessing"
    - "sklearn.compose"
    - "sklearn.pipeline"
    - "sklearn.linear_model"
    - "sklearn.ensemble"
    - "sklearn.metrics"

  pipeline_steps:
    - step: "Feature engineering"
      type: "Custom function"
      output_features:
        - "Total Spend"
        - "R&D Ratio"
        - "Marketing Ratio"
        - "Administration Ratio"
        - "Log Marketing Spend"
        - "R&D x Marketing"
    - step: "ColumnTransformer"
      numeric_transformer: "StandardScaler"
      categorical_transformer: "OneHotEncoder(drop='first', handle_unknown='ignore')"
    - step: "Model"
      options:
        - "LinearRegression"
        - "Ridge"
        - "Lasso"
        - "RandomForestRegressor"

experiments:
  - id: "E1"
    name: "Original features only"
    features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "State"
    purpose: "Baseline comparison."

  - id: "E2"
    name: "Remove Administration"
    features:
      - "R&D Spend"
      - "Marketing Spend"
      - "State"
    purpose: "Test if Administration is noise."

  - id: "E3"
    name: "Add spending ratio features"
    features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "State"
      - "Total Spend"
      - "R&D Ratio"
      - "Marketing Ratio"
      - "Administration Ratio"
    purpose: "Test whether budget allocation structure improves explanation."

  - id: "E4"
    name: "Marketing diminishing return"
    features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "Log Marketing Spend"
      - "State"
    purpose: "Test nonlinear marketing effect."

  - id: "E5"
    name: "Interaction effect"
    features:
      - "R&D Spend"
      - "Marketing Spend"
      - "R&D x Marketing"
      - "State"
    purpose: "Test whether R&D and Marketing reinforce each other."

final_recommendation_template:
  conclusion:
    - "Use Linear Regression or Ridge as the main model because the dataset is small and interpretation is important."
    - "Use Random Forest only as a supporting model for feature importance."
    - "Do not overclaim causality."
    - "Focus the final discussion on budget allocation strategy."
  business_message: >
    The model should be used to understand spending patterns, not to make perfect
    profit predictions. R&D investment is expected to be the strongest driver of
    Profit, Marketing may support growth, Administration should be tested as a
    possible weak feature, and State should be interpreted carefully as a possible
    proxy rather than a direct cause.