"""Data Analyst Agent — fully LLM-driven cleaning and analysis.

Generates step2_analysis.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from ada import paths as _paths
from ada.agents.llm_client import client, MODEL
from ada.agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert Data Analyst Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable.
- ALWAYS start the script with: import pandas as pd; import numpy as np; import json; import os
- Use pandas, numpy, matplotlib (Agg backend), seaborn, scipy — all available.
- Save plots as PNG files — never display them. Close each figure with plt.close().
- Make all preprocessing and analysis decisions based on the understanding outputs provided.
  Do NOT apply a fixed recipe — look at what the data actually needs.
- NEVER use nested ternary expressions (x if a else y if b else z ...).
  For ANY conditional logic with more than 2 branches, always use an if/elif/else block.
- NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
- NEVER use select_dtypes(include=['object']) — deprecated in pandas 3.x.
  Use: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
- NEVER use inplace=True on a column slice — pandas 3.x Copy-on-Write raises ChainedAssignmentError.
  WRONG: df[col].fillna(val, inplace=True)
  RIGHT: df[col] = df[col].fillna(val)
- NEVER use pd.np — removed in pandas 2.0. Always use numpy directly.
- NEVER use df.apply() — it triggers "Maximum recursion level reached" on DataFrames with object/string columns.
  WRONG: df.apply(lambda col: col.fillna(col.mode()[0]), axis=0)
  WRONG: df.apply(lambda row: ..., axis=1)
  RIGHT: use explicit for loops:
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
- NEVER use comparison operators (==, !=) on whole DataFrames with object columns.
  To compare a column, use: df[col] == value  (single column only, never df == value).
- For df.corr(), always pass numeric_only=True to avoid errors on non-numeric columns.
- Use EXACTLY this try/except structure — nothing outside it:

    try:
        # ... all computation, cleaning, plotting ...
        print(f"CLEANED_CSV:{cleaned_path}")
        print(json.dumps(analysis, indent=2, default=str))
    except Exception as e:
        import sys
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

  Both print statements MUST be the last two statements inside the try block.
  The except block MUST call sys.exit(1) so the runner detects failure.
  Do NOT add any statement after the except block.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
ALWAYS ensure the script starts with: import pandas as pd; import numpy as np; import json; import os
NEVER use pd.np — removed in pandas 2.0. Use numpy directly.
NEVER use nested ternary expressions — use if/elif/else blocks for any logic with more than 2 branches.
NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
NEVER use select_dtypes(include=['object']) — use: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
NEVER use inplace=True on a column slice — use: df[col] = df[col].fillna(val)
For df.corr(), always pass numeric_only=True.
"Maximum recursion level reached" FIX: Remove ALL df.apply() calls — they always cause this error on
DataFrames with object/string columns. Replace with explicit for loops:
  WRONG: df.apply(lambda col: col.fillna(col.mode()[0]), axis=0)
  RIGHT:
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
Also remove any whole-DataFrame comparison (df == value) — use df[col] == value per column.
KeyError on column name: the column may have been renamed or dropped earlier. Use df.columns.tolist()
to check available columns before accessing. Use df.get(col) or check 'col in df.columns' first.
The output structure MUST follow this EXACT pattern — both prints inside try, nothing after except:

    try:
        # ... all computation, cleaning, plotting ...
        print(f"CLEANED_CSV:{cleaned_path}")
        print(json.dumps(analysis, indent=2, default=str))
    except Exception as e:
        import sys
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

Do NOT add any statement after the except block."""


def generate_analysis_code(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    general_output: str,
    profiling_output: str,
    session_id: str = "",
) -> str:
    session_key  = session_id or "default"
    plots_dir    = str(_paths.OUTPUTS_DIR / session_key / "plots").replace("\\", "/")
    cleaned_path = str(_paths.UPLOADS_DIR / f"{session_key}_cleaned.csv").replace("\\", "/")
    dataset_path = dataset_path.replace("\\", "/")

    prompt = f"""Write Python code to clean a dataset and produce an analysis report.

Dataset: {dataset_path}
Task type: {task_type}
Target column: {target_column if target_column else "None (unsupervised)"}
Plots output directory: {plots_dir}
Cleaned CSV output path: {cleaned_path}

════════════════════════════════════════════════════════════════════
GENERAL UNDERSTANDING (universal metrics — column types, missing values,
correlations, outliers, encoding recommendations, missing value strategy)
════════════════════════════════════════════════════════════════════
{general_output[:4000]}

════════════════════════════════════════════════════════════════════
TASK-SPECIFIC PROFILE ({task_type} metrics — separability, target
distribution, variance ranking, imbalance, etc.)
════════════════════════════════════════════════════════════════════
{profiling_output[:3000]}

════════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════════

Read the two understanding outputs above carefully. Then write Python code that:

── STEP 1: LOAD & CLEAN ───────────────────────────────────────────────────
Load: df = pd.read_csv("{dataset_path}")
Record original_shape = df.shape

Apply the cleaning decisions the understanding agents diagnosed.
Use missing_value_strategy, outlier_analysis, and column_profiles to decide:
  • Drop columns classified as constant, id_like, free_text, or >50% missing
  • Fill missing values per column using the recommended strategy (median/mean/mode)
  • Cap outliers where recommended_action == "cap" (clip to 1st–99th percentile)
  • Parse datetime columns: extract year, month, day, dayofweek, hour; drop original
  • Drop exact duplicate rows

Do NOT apply categorical encoding (label/onehot/frequency) — the ML engineer does that.
Record every decision in preprocessing_decisions = {{}} dict.

── STEP 2: VISUALISATIONS ─────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
os.makedirs("{plots_dir}", exist_ok=True)

Parse both suggestion lists from the two understanding outputs shown above.
Implement ALL suggestions that are feasible on the cleaned DataFrame.
After every plt.savefig(...), call plt.close(). Append each filename to plots_saved.

── STEP 3: SAVE CLEANED CSV ───────────────────────────────────────────────
import os; os.makedirs(os.path.dirname("{cleaned_path}"), exist_ok=True)
df.to_csv("{cleaned_path}", index=False)
print(f"CLEANED_CSV:{cleaned_path}")   # exact marker, no spaces around colon

── STEP 4: OUTPUT ANALYSIS JSON ───────────────────────────────────────────
Required top-level keys:
  original_shape, cleaned_shape, col_types, encoding_recommendations,
  high_correlation_pairs, outlier_summary, preprocessing_decisions,
  class_distribution, imbalance_ratio, imbalance_recommendation,
  feature_target_correlations, task_specific, top_features,
  feature_insights, recommendations, plots_saved, plot_insights,
  ml_model_recommendations

The last two statements inside the try block MUST be (in this order):
  print(f"CLEANED_CSV:{cleaned_path}")
  print(json.dumps(analysis, indent=2, default=str))

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    if response.usage:
        _tok("data_analyst", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Data Analyst Agent generated code (session=%s).", session_key)
    return code


def fix_analysis_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("Data Analyst Agent fixing code (attempt %d)...", attempt)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
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
        _tok("data_analyst", response.usage.prompt_tokens, response.usage.completion_tokens)
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_analysis_code(current_code_path.read_text(encoding="utf-8"), stderr, stdout, attempt)
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
    profiling_output: str,
    session_id: str = "",
) -> dict:
    code = generate_analysis_code(
        dataset_path, task_type, target_column,
        general_output, profiling_output, session_id,
    )
    script_path = _paths.GENERATED_CODE_DIR / "step2_analysis.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step2_analysis.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
