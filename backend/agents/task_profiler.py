"""Task-Specific Profiling Agent — generates step1b_profiling.py.

Takes the general understanding output and computes task-specific metrics:
  - Classification: class distribution, imbalance, feature separability
  - Regression: target distribution, log-transform hint, feature-target correlations, heteroscedasticity
  - Clustering: variance ranking, scaling sensitivity, PCA dimensionality, k-range suggestion
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"

from agents.llm_client import client, MODEL
from agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert Data Profiling Agent. Your ONLY job is to write clean, executable Python code.
Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- Analyse the ENTIRE dataset — never sample for computation.
- Handle missing values before any computation: fillna(median) for numeric, fillna(mode()[0]) for categorical.
- Round all floats to 4 decimal places.
- NEVER use nested ternary expressions (x if a else y if b else z ...).
  For ANY conditional logic with more than 2 branches, always use an if/elif/else block.
- NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
- For df.corr(), always pass numeric_only=True to avoid errors on non-numeric columns.
- NEVER call .std(), .var(), .mean() on the full DataFrame directly.
  Always isolate numeric columns first: numeric_df = df.select_dtypes(include=[np.number])
  Then aggregate on numeric_df. Use .index.tolist() on the result — never index df.columns
  with a mask from a numeric subset (lengths differ → "Boolean index has wrong length").
- Use EXACTLY this try/except structure — nothing outside it:

    try:
        # ... all computation ...
        result = { ... }
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        import sys
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

  The print(json.dumps(result, ...)) MUST be the last statement inside the try block.
  The except block MUST call sys.exit(1) so the runner detects failure.
  Do NOT add any statement after the except block.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
NEVER use nested ternary expressions — use if/elif/else blocks for any logic with more than 2 branches.
NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
For df.corr(), always pass numeric_only=True.
NEVER call .std(), .var(), .mean() on the full DataFrame — filter to numeric first:
    numeric_df = df.select_dtypes(include=[np.number]); vals = numeric_df.std()
NEVER index df.columns with a mask from a numeric subset — use .index.tolist() instead.
The output structure MUST follow this EXACT pattern — the print MUST be the last statement inside try:

    try:
        # ... all computation ...
        result = { ... }
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        import sys
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

Do NOT add any statement after the except block."""


def _classification_requirements(target_column: str) -> str:
    return f"""
TASK: supervised_classification  |  Target column: {target_column}

Compute ALL of the following and store in result dict:

1. class_distribution
   dict {{class_label: count}} for every unique value in "{target_column}"

2. class_balance_pct
   dict {{class_label: float}} — each class count as a fraction of total rows (4dp)

3. imbalance_ratio
   float — max_class_count / min_class_count (4dp)

4. imbalance_action
   str — one of: "smote" | "class_weight='balanced'" | "none"
   Rules: imbalance_ratio > 10 AND min_class_count < 100 → "smote"
          3 < imbalance_ratio ≤ 10 → "class_weight='balanced'"
          imbalance_ratio ≤ 3 → "none"

5. imbalance_reason
   str — brief explanation of which threshold triggered the action

6. feature_separability
   dict {{col: float}} — for each NUMERIC column (excluding target):
   ratio = between_class_variance / total_variance
   between_class_variance: variance of the series where each row is replaced by its class mean for that column
   total_variance: df[col].var()
   Skip columns where total_variance == 0. Round to 4dp.
   Sort descending by ratio.

7. top_discriminative_features
   list — top 8 column names sorted by feature_separability (highest first)

8. numeric_cols_used
   list — all numeric column names used in the separability computation

9. suggested_visualizations
   List of 3–5 classification-specific plots grounded in the metrics you computed.

   AVAILABLE PLOT TYPES for classification (reference menu — reasoning required):
     count_plot, kde_by_class, box_by_class, violin_by_class,
     scatter_colored_by_class, pairplot_with_class_hue, pca_scatter,
     feature_separability_bar, stacked_bar_categorical_vs_class, swarm_by_class

   For EVERY suggestion:
   - trigger_finding: cite the exact metric value that motivates this plot
     e.g. "imbalance_ratio=15.3, minority class has 47 samples",
          "top separability feature col_X has ratio=0.82"
   - reason: what modeling problem or data characteristic this makes visible
   Format: {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

Required result keys:
  class_distribution, class_balance_pct, imbalance_ratio, imbalance_action,
  imbalance_reason, feature_separability, top_discriminative_features,
  numeric_cols_used, suggested_visualizations"""


def _regression_requirements(target_column: str) -> str:
    return f"""
TASK: supervised_regression  |  Target column: {target_column}

Compute ALL of the following and store in result dict:

1. target_stats
   dict — mean, std, min, max, median, skewness, kurtosis of "{target_column}" (4dp each)

2. log_transform_suggested
   bool — True if abs(skewness) > 1.0 AND all target values > 0

3. target_outlier_pct
   float — fraction of target values outside IQR bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (4dp)

4. feature_target_correlations
   dict {{col: float}} — Pearson r between each NUMERIC feature column and "{target_column}".
   KEY must be ONLY the feature column name — never "feat_target" or any compound name.
   Sort descending by abs(r). Round to 4dp. Exclude "{target_column}" itself.

5. top_predictive_features
   list — top 8 feature names by abs(r) with target

6. heteroscedasticity_hint
   dict {{col: float}} — for each numeric feature col, Pearson correlation between that feature
   and abs(target - target.mean()). Sort descending by abs value. Round to 4dp.

7. heteroscedastic_features
   list — top 5 column names from heteroscedasticity_hint by abs value

8. suggested_visualizations
   List of 3–5 regression-specific plots grounded in the metrics you computed.

   AVAILABLE PLOT TYPES for regression (reference menu — reasoning required):
     target_histogram_kde, scatter_feature_vs_target, pair_plot,
     residual_vs_fitted, qq_plot, hexbin_plot, correlation_heatmap,
     feature_vs_abs_residual, log_transform_comparison, box_plot_outliers

   For EVERY suggestion:
   - trigger_finding: cite the exact metric value that motivates this plot
     e.g. "target skewness=2.8, log_transform_suggested=True",
          "col_X has feature_target_correlation=0.79",
          "col_Y leads heteroscedasticity_hint with r=0.65"
   - reason: what modeling risk or relationship this exposes
   Format: {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

Required result keys:
  target_stats, log_transform_suggested, target_outlier_pct, feature_target_correlations,
  top_predictive_features, heteroscedasticity_hint, heteroscedastic_features,
  suggested_visualizations"""


def _clustering_requirements() -> str:
    return """
TASK: unsupervised_clustering  |  No target column

Compute ALL of the following and store in result dict:

1. feature_variance_ranking
   dict {{col: float}} — variance of each NUMERIC column. Sort descending. Round to 4dp.

2. top_variance_features
   list — top 8 column names by variance

3. scaling_sensitivity
   float — max_std / min_std across all numeric columns (4dp).
   If min_std == 0 set to 999.0.

4. scaling_required
   bool — True if scaling_sensitivity > 5.0

5. pca_components_for_90pct_variance
   int — number of PCA components (fit on StandardScaler-transformed numeric data)
   needed to explain ≥90% cumulative variance.
   Cap n_components at min(n_numeric_cols, n_rows, 15).

6. pca_variance_ratios
   list of floats — explained_variance_ratio_ for first 8 components (4dp each)

7. suggested_k_range
   list [min_k, max_k] — min_k=2, max_k=min(int(sqrt(n_rows)), 15)

8. categorical_cols_present
   list — names of non-numeric columns in the dataset
   (distance-based algorithms cannot handle these natively)

9. n_numeric_cols
   int — count of numeric columns

10. n_rows
    int — total row count

11. suggested_visualizations
    List of 3–5 clustering-specific plots grounded in the metrics you computed.

    AVAILABLE PLOT TYPES for clustering (reference menu — reasoning required):
      pca_scatter, feature_variance_bar, correlation_heatmap,
      pairplot_top_features, box_plot_by_variance, density_heatmap,
      elbow_curve_k_range, distance_matrix_heatmap, scaling_spread_bar

    For EVERY suggestion:
    - trigger_finding: cite the exact metric value that motivates this plot
      e.g. "scaling_sensitivity=45.2, scaling_required=True",
           "pca_components_for_90pct=3, first 3 PCs explain 91% variance",
           "top variance feature col_X variance=12500.3"
    - reason: what cluster structure, scaling issue, or density pattern this reveals
    Format: {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

Required result keys:
  feature_variance_ranking, top_variance_features, scaling_sensitivity, scaling_required,
  pca_components_for_90pct_variance, pca_variance_ratios, suggested_k_range,
  categorical_cols_present, n_numeric_cols, n_rows, suggested_visualizations"""


def generate_task_profiling_code(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    general_output: str,
) -> str:
    is_classif = "classif" in task_type
    is_regress = "regress" in task_type

    if is_classif and target_column:
        task_requirements = _classification_requirements(target_column)
    elif is_regress and target_column:
        task_requirements = _regression_requirements(target_column)
    else:
        task_requirements = _clustering_requirements()

    prompt = f"""Write executable Python code to compute task-specific profiling metrics.

Dataset: {dataset_path}
Task type: {task_type}
Target column: {target_column if target_column else "None (unsupervised)"}

=== GENERAL UNDERSTANDING (use to know column types, which columns are numeric, etc.) ===
{general_output[:3000]}
=== END GENERAL UNDERSTANDING ===

{task_requirements}

IMPLEMENTATION RULES:
- Load the dataset fresh: pd.read_csv("{dataset_path}")
- Analyse the ENTIRE dataset
- Fill missing values before computation: numeric → fillna(median), categorical → fillna(mode()[0])
- Round ALL floats to 4 decimal places
- Use EXACTLY this structure — result and print MUST be inside the try block:

    try:
        # ... all computation ...
        result = {{ ... }}   # all required keys
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        import sys
        print(json.dumps({{"error": str(e)}}, indent=2))
        sys.exit(1)

  Do NOT add any statement after the except block.

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    if response.usage:
        _tok("task_profiler", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Task Profiling Agent generated code for %s.", task_type)
    return code


def fix_profiling_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("Task Profiling Agent fixing code (attempt %d)...", attempt)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user",   "content": f"""Code failed.

ERROR:
{stderr}

STDOUT:
{stdout}

CODE:
{current_code}

Fix it. Output ONLY the complete fixed Python code."""},
        ],
    )
    if response.usage:
        _tok("task_profiler", response.usage.prompt_tokens, response.usage.completion_tokens)
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_profiling_code(current_code_path.read_text(), stderr, stdout, attempt)
    return callback


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    general_output: str,
) -> dict:
    logger.info("Step 1b: Generating task-specific profiling code for %s...", task_type)
    code = generate_task_profiling_code(dataset_path, task_type, target_column, general_output)

    script_path = GENERATED_CODE_DIR / "step1b_profiling.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step1b_profiling.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
