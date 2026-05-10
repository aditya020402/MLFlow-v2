"""Task-Specific Profiling Agent — generates step1b_profiling.py."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from ada import paths as _paths
from ada.agents.llm_client import client, MODEL
from ada.agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert Data Profiling Agent. Your ONLY job is to write clean, executable Python code.
Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- ALWAYS start the script with: import pandas as pd; import numpy as np; import json
- Analyse the ENTIRE dataset — never sample for computation.
- Handle missing values before any computation: fillna(median) for numeric, fillna(mode()[0]) for categorical.
- Round all floats to 4 decimal places.
- NEVER use nested ternary expressions (x if a else y if b else z ...).
  For ANY conditional logic with more than 2 branches, always use an if/elif/else block.
- NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
- NEVER pass pd.api.types.is_numeric_dtype as a dtype argument to select_dtypes() — it is a function call, not a dtype.
  WRONG: df.select_dtypes(include=[pd.api.types.is_numeric_dtype])
  RIGHT: numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
- For df.corr(), always pass numeric_only=True to avoid errors on non-numeric columns.
- NEVER call .std(), .var(), .mean() on the full DataFrame directly.
  Always isolate numeric columns first: numeric_df = df.select_dtypes(include=[np.number])
  Then aggregate on numeric_df. Use .index.tolist() on the result — never index df.columns
  with a mask from a numeric subset (lengths differ → "Boolean index has wrong length").
- NEVER use select_dtypes(include=['object']) to find categorical/string columns — deprecated in pandas 3.x.
  Use: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
- NEVER use inplace=True on a column slice — pandas 3.x Copy-on-Write raises ChainedAssignmentError.
  WRONG: df[col].fillna(val, inplace=True)
  RIGHT: df[col] = df[col].fillna(val)
- NEVER use pd.np — it was removed in pandas 2.0. Always use numpy directly: import numpy as np
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
ALWAYS ensure the script starts with: import pandas as pd; import numpy as np; import json
NEVER use pd.np — removed in pandas 2.0. Use numpy directly.
NEVER use nested ternary expressions — use if/elif/else blocks for any logic with more than 2 branches.
NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
NEVER pass pd.api.types.is_numeric_dtype as a dtype to select_dtypes() — it is a callable, not a dtype.
  WRONG: df.select_dtypes(include=[pd.api.types.is_numeric_dtype])
  RIGHT: numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
For df.corr(), always pass numeric_only=True.
NEVER call .std(), .var(), .mean() on the full DataFrame — filter to numeric first:
    numeric_df = df.select_dtypes(include=[np.number]); vals = numeric_df.std()
NEVER index df.columns with a mask from a numeric subset — use .index.tolist() instead.
PANDAS 3.x COMPATIBILITY: NEVER use select_dtypes(include=['object']) to find categorical/string columns.
Use this instead: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
COPY-ON-WRITE FIX: NEVER use inplace=True on a column slice — causes ChainedAssignmentError in pandas 3.x.
  WRONG: df[col].fillna(val, inplace=True)
  RIGHT: df[col] = df[col].fillna(val)
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
   Sort descending by ratio.

7. top_discriminative_features
   list — top 8 column names sorted by feature_separability (highest first)

8. numeric_cols_used
   list — all numeric column names used in the separability computation

9. suggested_visualizations
   List of 3–5 classification-specific plots grounded in the metrics you computed.
   Format: {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

Required result keys:
  class_distribution, class_balance_pct, imbalance_ratio, imbalance_action,
  imbalance_reason, feature_separability, top_discriminative_features,
  numeric_cols_used, suggested_visualizations"""


def _regression_requirements(target_column: str) -> str:
    return f"""
TASK: supervised_regression  |  Target column: {target_column}

Compute ALL of the following and store in result dict:

1. target_stats — mean, std, min, max, median, skewness, kurtosis (4dp each)
2. log_transform_suggested — bool
3. target_outlier_pct — float
4. feature_target_correlations — dict {{col: float}}, sort descending by abs(r)
5. top_predictive_features — list, top 8
6. heteroscedasticity_hint — dict {{col: float}}
7. heteroscedastic_features — list, top 5
8. suggested_visualizations — 3–5 plots
   Format: {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

Required result keys:
  target_stats, log_transform_suggested, target_outlier_pct, feature_target_correlations,
  top_predictive_features, heteroscedasticity_hint, heteroscedastic_features,
  suggested_visualizations"""


def _clustering_requirements() -> str:
    return """
TASK: unsupervised_clustering  |  No target column

Compute ALL of the following and store in result dict:

1. feature_variance_ranking — dict {{col: float}}, sort descending
2. top_variance_features — list, top 8
3. scaling_sensitivity — float (max_std / min_std)
4. scaling_required — bool (True if > 5.0)
5. pca_components_for_90pct_variance — int
6. pca_variance_ratios — list of floats (first 8 components)
7. suggested_k_range — [min_k, max_k]
8. categorical_cols_present — list
9. n_numeric_cols — int
10. n_rows — int
11. suggested_visualizations — 3–5 plots
    Format: {"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}

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
    dataset_path = dataset_path.replace("\\", "/")
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
- Use EXACTLY this structure:

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
        return fix_profiling_code(current_code_path.read_text(encoding="utf-8"), stderr, stdout, attempt)
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

    script_path = _paths.GENERATED_CODE_DIR / "step1b_profiling.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step1b_profiling.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
