"""Data Analyst Agent — fully LLM-driven cleaning and analysis.

Receives outputs from both understanding agents and decides all techniques
autonomously based on what the data actually looks like.
Generates step2_analysis.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR        = Path(__file__).parent.parent / "outputs"
UPLOADS_DIR        = Path(__file__).parent.parent / "uploads"

from agents.llm_client import client, MODEL
from agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert Data Analyst Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable.
- Use pandas, numpy, matplotlib (Agg backend), seaborn, scipy — all available.
- Save plots as PNG files — never display them. Close each figure with plt.close().
- Make all preprocessing and analysis decisions based on the understanding outputs provided.
  Do NOT apply a fixed recipe — look at what the data actually needs.
- NEVER use nested ternary expressions (x if a else y if b else z ...).
  For ANY conditional logic with more than 2 branches, always use an if/elif/else block.
- NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
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
NEVER use nested ternary expressions — use if/elif/else blocks for any logic with more than 2 branches.
NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
For df.corr(), always pass numeric_only=True.
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
    plots_dir    = str(OUTPUTS_DIR / session_key / "plots")
    cleaned_path = str(UPLOADS_DIR / f"{session_key}_cleaned.csv")

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

Both understanding agents embedded plot suggestions in their JSON outputs:
  • GENERAL UNDERSTANDING JSON  → key "suggested_exploratory_plots"  (universal findings)
  • TASK-SPECIFIC PROFILING JSON → key "suggested_visualizations"    (task-specific findings)

Parse both suggestion lists from the two understanding outputs shown above.
Implement ALL suggestions that are feasible on the cleaned DataFrame.

Each suggestion has: plot_type, features, trigger_finding, reason.
Use plot_type and features to choose the right matplotlib/seaborn call.
Add a Python comment above each plot block: # Triggered by: <trigger_finding>

If a suggested feature no longer exists after cleaning, skip it gracefully.
After every plt.savefig(...), call plt.close(). Append each filename to plots_saved.

── STEP 3: SAVE CLEANED CSV ───────────────────────────────────────────────
df.to_csv("{cleaned_path}", index=False)
print(f"CLEANED_CSV:{cleaned_path}")   # exact marker, no spaces around colon

── STEP 4: OUTPUT ANALYSIS JSON ───────────────────────────────────────────
Assemble an analysis dict and print it as JSON.

Required top-level keys (fill irrelevant ones with {{}} / [] / "n/a"):
  original_shape          — list [rows, cols] before cleaning
  cleaned_shape           — list [rows, cols] after cleaning
  col_types               — dict {{col: inferred_type}} from the general understanding
  encoding_recommendations — dict from the general understanding
  high_correlation_pairs  — list from the general understanding
  outlier_summary         — dict {{col: {{pct_outliers, recommended_action}}}} from general understanding
  preprocessing_decisions — dict you built in Step 1
  class_distribution      — dict (classification) or {{}} (other)
  imbalance_ratio         — float (classification) or 1.0 (other)
  imbalance_recommendation — str (classification) or "n/a" (other)
  feature_target_correlations — dict (regression/classification) or {{}} (clustering)
  task_specific           — dict with the key task-specific findings from the profiling output
  top_features            — list of most informative features for this task
  feature_insights        — list of 3–5 key insights about this data
  recommendations         — list of 3–5 actionable preprocessing recommendations
  plots_saved             — list of plot filenames actually saved
  plot_insights           — list of 3–5 observations derived from the plot findings and
                            profiling statistics; each entry should state a specific pattern,
                            risk, or anomaly and its implication for modeling
                            (e.g. "feature_X shows bimodal distribution split by class —
                            high discriminative power, tree splits will exploit this";
                            "target is right-skewed with skewness=2.8 — log-transform
                            will stabilise variance for linear models")
  ml_model_recommendations — dict with these keys:
      "algorithm_hints"              : list — which algorithm families suit this data and why,
                                       grounded in the profiling metrics
      "preprocessing_notes"          : list — any remaining preprocessing the ML engineer must do
      "expected_challenges"          : list — specific risks the ML engineer should handle
                                       (imbalance, collinearity, outliers, high dimensionality, etc.)
      "feature_importance_candidates": list — top features most likely to matter, ranked

The last two statements inside the try block MUST be (in this order):
  print(f"CLEANED_CSV:{cleaned_path}")
  print(json.dumps(analysis, indent=2, default=str))

These prints go INSIDE the try block — NOT after it. The except block must call sys.exit(1).
Do NOT add any statement after the except block.

IMPORTANT:
- Base ALL decisions on the two understanding outputs — do not use a fixed recipe
- Analyse the entire dataset
- Use real variable names, not pseudocode

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
        return fix_analysis_code(current_code_path.read_text(), stderr, stdout, attempt)
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
    script_path = GENERATED_CODE_DIR / "step2_analysis.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step2_analysis.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
