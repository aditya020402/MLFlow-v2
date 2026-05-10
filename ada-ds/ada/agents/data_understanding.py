"""General Data Understanding Agent — computes universal dataset metrics.

Generates step1a_general.py, which outputs a structured JSON with:
  dataset_overview, column_profiles, correlation_analysis, outlier_analysis,
  encoding_recommendations, missing_value_strategy, llm_insights
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from ada import paths as _paths
from ada.agents.llm_client import client, MODEL
from ada.agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert Data Understanding Agent. Your ONLY job is to write clean, executable Python code.
Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable with pandas, numpy, sklearn available.
- NEVER sample the dataset — always analyse the ENTIRE dataset.
- All float values in the output must be rounded to 4 decimal places.
- NEVER use nested ternary expressions (x if a else y if b else z ...).
  For ANY conditional logic with more than 2 branches, always use an if/elif/else block.
- NEVER use np.issubdtype(col.dtype, np.number) — it crashes on pandas StringDtype.
  Always use pd.api.types.is_numeric_dtype(df[col]) to check if a column is numeric.
- For df.corr(), always pass numeric_only=True to avoid errors on non-numeric columns.
- NEVER call .std(), .var(), .mean(), or any aggregation on the full DataFrame directly.
  Always isolate numeric columns first, then aggregate:
      numeric_df = df.select_dtypes(include=[np.number])
      std_vals   = numeric_df.std()
      near_zero  = std_vals[std_vals < 0.001].index.tolist()   # use .index.tolist()
- NEVER index df.columns with a boolean mask derived from a numeric-only subset.
  A mask from select_dtypes(...).std() has fewer entries than df.columns — applying it
  to df.columns raises "Boolean index has wrong length".
  ALWAYS use .index.tolist() on the subset result, never df.columns[mask].
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

PREVIEW_SYSTEM_PROMPT = """You are a data science expert reviewing a dataset sample.
Respond ONLY with a valid JSON object — no markdown, no backticks, no extra text."""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
NEVER use nested ternary expressions — use if/elif/else blocks for any logic with more than 2 branches.
NEVER use np.issubdtype(col.dtype, np.number) — use pd.api.types.is_numeric_dtype(df[col]) instead.
For df.corr(), always pass numeric_only=True.
NEVER call .std(), .var(), .mean() on the full DataFrame — always filter to numeric columns first:
    numeric_df = df.select_dtypes(include=[np.number])
    std_vals   = numeric_df.std()
    near_zero  = std_vals[std_vals < 0.001].index.tolist()
NEVER index df.columns with a boolean mask from a numeric subset — lengths differ and this raises
"Boolean index has wrong length". Use .index.tolist() on the subset result instead.
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


def preview_dataset_context(dataset_path: str, task_type: str, target_column: str = None) -> dict:
    dataset_path = dataset_path.replace("\\", "/")
    try:
        import pandas as pd
        df_sample = pd.read_csv(dataset_path, nrows=1000)
    except Exception as exc:
        logger.warning("Could not read dataset preview: %s", exc)
        return {}

    first_20_csv = df_sample.head(20).to_csv(index=False)
    dtypes_str   = df_sample.dtypes.to_string()
    nunique_str  = df_sample.nunique().to_string()
    describe_str = df_sample.describe(include="all").to_string()
    target_line  = (f"Target column (to predict): {target_column}"
                    if target_column else "No target column (unsupervised task)")

    prompt = f"""Dataset: {dataset_path}
Task type: {task_type}
{target_line}

=== First 20 rows (CSV) ===
{first_20_csv}

=== Column dtypes ===
{dtypes_str}

=== Unique value counts ===
{nunique_str}

=== Descriptive statistics (first 1000 rows) ===
{describe_str}

Based on the sample above, return this JSON:
{{
    "dataset_description": "<2-3 sentences: what this dataset represents and its domain>",
    "column_relationships": [
        "<logical or causal relationship between specific columns>"
    ],
    "feature_engineering_ideas": [
        "<concrete new feature derivable from existing columns>"
    ],
    "data_quality_concerns": [
        "<specific concern per column, e.g. suspicious zeros, extreme outliers>"
    ],
    "recommended_focus_areas": [
        "<what to pay most attention to for this task>"
    ],
    "target_column_notes": "<if target exists: describe its distribution, class balance, and challenges; else 'N/A'>"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system",  "content": PREVIEW_SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
        )
        if response.usage:
            _tok("data_understanding", response.usage.prompt_tokens, response.usage.completion_tokens)
        text = _strip_markdown(response.choices[0].message.content.strip())
        context = json.loads(text)
        logger.info("Dataset context preview done: %s", context.get("dataset_description", "")[:80])
        return context
    except Exception as exc:
        logger.warning("Could not parse dataset context: %s", exc)
        return {}


def generate_understanding_code(
    dataset_path: str,
    task_type: str,
    target_column: str = None,
    dataset_context: dict = None,
) -> str:
    dataset_path     = dataset_path.replace("\\", "/")
    dataset_context  = dataset_context or {}
    context_literal  = json.dumps(dataset_context, indent=4, default=str)

    prompt = f"""Write executable Python code to compute a structured universal profile of this dataset.

Dataset: {dataset_path}
Task type: {task_type}
Target column: {target_column if target_column else "None (unsupervised)"}

=== LLM PREVIEW INSIGHTS ===
(Embed this dict VERBATIM in your output JSON under the key "llm_insights")
{context_literal}
=== END PREVIEW INSIGHTS ===

Your code must load the ENTIRE dataset and compute a JSON with exactly these 8 top-level keys:

── 1. dataset_overview ──────────────────────────────────────────────────────
  n_rows: int, n_cols: int, duplicate_rows: int,
  total_missing_pct: float (fraction 0–1, 4dp), memory_mb: float (4dp)

── 2. column_profiles  (dict: column_name → profile dict) ──────────────────
  For EVERY column compute:
  • inferred_type  — use this EXACT if/elif/else structure (no nested ternaries):

    def _infer_type(col, df, n_rows):
        s = df[col]
        if s.nunique() == 1:
            return "constant"
        if s.dtype == bool or str(s.dtype) == 'bool':
            return "boolean"
        if s.dtype == 'object' or str(s.dtype) == 'category':
            try:
                parsed = pd.to_datetime(s, errors='coerce')
                if parsed.notna().mean() >= 0.70:
                    return "datetime"
            except Exception:
                pass
            mean_len = s.dropna().astype(str).str.len().mean()
            if mean_len > 40 or s.nunique() / n_rows > 0.5:
                return "free_text"
            if s.nunique() <= 2:
                return "boolean"
            if s.nunique() <= 10:
                return "categorical_low"
            return "categorical_high"
        # numeric branch
        if s.nunique() == n_rows and str(s.dtype).startswith('int'):
            if s.is_monotonic_increasing:
                return "id_like"
        return "numeric"

    Use this helper (copy it into your script) and call it for every column.
  • missing_pct: float (fraction 0–1, NOT a percentage)
  • nunique: int
  • stats (numeric cols only): dict with mean, std, min, max, median, skewness, kurtosis (all 4dp)
  • top_values (categorical cols only): dict of top-5 {{value: count}}
  • is_ordinal_hint: bool — True if values match patterns like
    low/med/high, small/medium/large, 1/2/3, bad/ok/good, bronze/silver/gold
  • looks_like_id: bool

── 3. correlation_analysis ──────────────────────────────────────────────────
  • high_correlation_pairs: list of {{"pair": "colA||colB", "r": float}} where |r| > 0.90
    (upper triangle only — no self-pairs, no duplicates)
  • near_zero_variance_cols: list of numeric column names where std < 0.001.
    Compute with this EXACT pattern (never apply to all columns, never index df.columns with a mask):
        _num = df.select_dtypes(include=[np.number])
        _std = _num.std()
        near_zero_variance_cols = _std[_std < 0.001].index.tolist()

── 4. outlier_analysis  (numeric columns only, IQR method) ──────────────────
  Per column: {{"pct_outliers": float, "recommended_action": "cap|keep", "reason": str}}
  Rules: pct_outliers < 0.05 OR > 0.30 → "keep";  0.05 ≤ pct ≤ 0.30 → "cap"

── 5. encoding_recommendations  (categorical + datetime columns only) ────────
  Per column: {{"strategy": str, "reason": str}}
  Strategy values: label | onehot | frequency | target_encode | drop | datetime_extract
  Rules:
    inferred_type == "datetime"                   → datetime_extract
    categorical, nunique ≤ 10                     → label
    categorical, 10 < nunique ≤ 20               → onehot
    categorical, 20 < nunique ≤ 100              → frequency
    categorical, nunique > 100 + classification  → target_encode
    categorical, nunique > 100 + other           → drop

── 6. missing_value_strategy  (columns where missing_pct > 0 only) ──────────
  Per column: {{"pct_missing": float, "recommended_strategy": str, "reason": str}}
  Strategy values: median | mean | mode | drop_column
  Rules:
    pct_missing > 0.5                → drop_column
    numeric + |skewness| > 1        → median
    numeric                          → mean
    categorical                      → mode

── 7. llm_insights ──────────────────────────────────────────────────────────
  Embed the LLM preview insights dict verbatim.

── 8. suggested_exploratory_plots ───────────────────────────────────────────
  3–5 universal exploratory plots grounded in findings you computed in keys 1–6.

  AVAILABLE PLOT TYPES (reference menu — do NOT pick blindly from this list):
    histogram, kde_plot, box_plot, violin_plot, scatter_plot, pair_plot,
    correlation_heatmap, missing_value_heatmap, missing_pct_bar,
    qq_plot, z_score_distribution

  For EVERY suggestion you MUST:
  1. Quote a specific computed finding that triggers this choice
  2. Name the exact feature(s) involved
  3. Explain what modeling problem or pattern this plot will make visible

  DO NOT suggest a plot you cannot justify with a specific computed value.
  Format per suggestion:
  {{"plot_type": str, "features": [str], "trigger_finding": str, "reason": str}}

IMPLEMENTATION RULES:
- Analyse the ENTIRE dataset — never call .sample() or .head() for computation
- Round ALL floats to 4 decimal places
- Use EXACTLY this structure:

    try:
        # ... all computation ...
        result = {{ ... }}   # all 8 keys
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
        _tok("data_understanding", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("General Understanding Agent generated code.")
    return code


def fix_understanding_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("General Understanding Agent fixing code (attempt %d)...", attempt)
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
        _tok("data_understanding", response.usage.prompt_tokens, response.usage.completion_tokens)
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_understanding_code(current_code_path.read_text(encoding="utf-8"), stderr, stdout, attempt)
    return callback


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(dataset_path: str, task_type: str, target_column: str = None) -> dict:
    logger.info("Step 1a: Previewing dataset for domain context...")
    dataset_context = preview_dataset_context(dataset_path, task_type, target_column)

    logger.info("Step 1a: Generating universal profiling code...")
    code = generate_understanding_code(dataset_path, task_type, target_column, dataset_context)

    script_path = _paths.GENERATED_CODE_DIR / "step1a_general.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step1a_general.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
