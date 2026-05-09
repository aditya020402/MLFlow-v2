import logging
import json
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert Data Understanding Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable with pandas, numpy, sklearn available.
- NEVER sample the dataset — always analyse the ENTIRE dataset.
- At the end print exactly ONE JSON object to stdout: print(json.dumps(summary, indent=2, default=str))
- Wrap the entire body in a try/except so the script never crashes silently.
- All float values in the summary must be rounded to 4 decimal places.
"""

PREVIEW_SYSTEM_PROMPT = """You are a data science expert reviewing a dataset sample.
Respond ONLY with a valid JSON object — no markdown, no backticks, no extra text."""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
IMPORTANT: keys in feature_target_correlations must be ONLY the feature column name (e.g. "age"),
never a doubled name like "age_age"."""


def preview_dataset_context(dataset_path: str, task_type: str, target_column: str = None) -> dict:
    """Read the first 1000 rows to let the LLM understand the dataset's business context.

    This lightweight step runs before code generation so the generated code can be
    more targeted — e.g. knowing which columns have logical relationships, what
    feature engineering is feasible, and what quality concerns to watch for.
    """
    try:
        import pandas as pd
        df_sample = pd.read_csv(dataset_path, nrows=1000)
    except Exception as exc:
        logger.warning("Could not read dataset preview: %s", exc)
        return {}

    first_20_csv   = df_sample.head(20).to_csv(index=False)
    dtypes_str     = df_sample.dtypes.to_string()
    nunique_str    = df_sample.nunique().to_string()
    describe_str   = df_sample.describe(include="all").to_string()
    target_line    = (f"Target column (to predict): {target_column}"
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
    "<logical or causal relationship between specific columns, e.g. 'height and weight together determine BMI'>",
    "..."
  ],
  "feature_engineering_ideas": [
    "<a concrete new feature derivable from existing columns, e.g. 'age_group (binned from age)', 'bmi = weight/height^2'>",
    "..."
  ],
  "data_quality_concerns": [
    "<specific concern per column, e.g. 'age column has value 0 which is suspicious', 'salary has extreme outliers'>",
    "..."
  ],
  "recommended_focus_areas": [
    "<what to pay most attention to for this task, e.g. 'class imbalance in target', 'multicollinearity between income and wealth'>",
    "..."
  ],
  "target_column_notes": "<if target exists: describe its distribution, class balance, and challenges; else 'N/A'>"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": PREVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content.strip()
        text = _strip_markdown(text)
        context = json.loads(text)
        logger.info("Dataset context preview completed: %s", context.get("dataset_description", "")[:80])
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
    """Generate a comprehensive EDA script using the code-skeleton approach.

    Covers the following EDA steps where applicable (based on the data):
      shape, preview records, dtypes, memory usage, missing values, duplicates,
      target distribution, feature type separation, per-column univariate stats
      (mean, std, min/max, quartiles, skewness, kurtosis, outlier count),
      categorical frequency analysis, inter-feature correlation matrix,
      feature-target correlations, and qualitative insights from the LLM preview.
    """
    dataset_context  = dataset_context or {}
    target_val       = f'"{target_column}"' if target_column else "None"
    is_classif       = "classif" in task_type
    is_regress       = "regress" in task_type

    # ── Inject LLM preview context as a Python literal ──────────────────────
    context_literal  = json.dumps(dataset_context, indent=4, default=str)

    # ── Build feature-target correlation block deterministically ────────────
    if target_column and (is_classif or is_regress):
        if is_classif:
            feat_corr_block = f"""\
from sklearn.preprocessing import LabelEncoder as _LE
_le = _LE()
_target_enc = pd.Series(
    _le.fit_transform(df["{target_column}"].astype(str).fillna("__missing__")),
    index=df.index,
)
_num_feats = [c for c in numeric_cols if c != "{target_column}"]
_raw_fc = {{}}
for _f in _num_feats:
    _s = df[_f].fillna(df[_f].median())
    _raw_fc[_f] = round(float(_s.corr(_target_enc)), 4)  # key = feature name only
feature_target_corr = dict(sorted(_raw_fc.items(), key=lambda x: abs(x[1]), reverse=True))"""
        else:
            feat_corr_block = f"""\
_target_s = df["{target_column}"].fillna(df["{target_column}"].median())
_num_feats = [c for c in numeric_cols if c != "{target_column}"]
_raw_fc = {{}}
for _f in _num_feats:
    _s = df[_f].fillna(df[_f].median())
    _raw_fc[_f] = round(float(_s.corr(_target_s)), 4)    # key = feature name only
feature_target_corr = dict(sorted(_raw_fc.items(), key=lambda x: abs(x[1]), reverse=True))"""
    else:
        feat_corr_block = "feature_target_corr = {}  # unsupervised — no target"

    # ── Target analysis block ────────────────────────────────────────────────
    if is_classif and target_column:
        target_analysis_block = f"""\
if target_column in df.columns:
    _vc = df[target_column].value_counts()
    _tdtype = str(df[target_column].dtype)
    target_analysis = {{
        "dtype":           _tdtype,
        "class_counts":    {{str(k): int(v) for k, v in _vc.items()}},
        "class_balance":   {{str(k): round(int(v) / num_rows * 100, 4) for k, v in _vc.items()}},
        "imbalance_ratio": round(float(_vc.iloc[0]) / float(_vc.iloc[-1]), 4) if len(_vc) > 1 else 1.0,
        "num_classes":     int(df[target_column].nunique()),
    }}
else:
    target_analysis = {{}}"""
    elif is_regress and target_column:
        target_analysis_block = f"""\
if target_column in df.columns:
    _ts = df[target_column].describe()
    target_analysis = {{
        "dtype":           str(df[target_column].dtype),
        "target_stats":    {{k: round(float(v), 4) for k, v in _ts.items()}},
        "target_skewness": round(float(df[target_column].skew()), 4),
        "target_kurtosis": round(float(df[target_column].kurt()), 4),
    }}
else:
    target_analysis = {{}}"""
    else:
        target_analysis_block = "target_analysis = {}  # unsupervised"

    prompt = f"""Write Python code to perform comprehensive data understanding on the dataset at:
  {dataset_path}

Task type: {task_type}
Target column: {target_column if target_column else "None (unsupervised)"}

The LLM has already previewed the first 1000 rows and produced this context — inject it verbatim
into the summary JSON as "dataset_context":
{context_literal}

Analyse the ENTIRE dataset — never call df.sample() or df.head() for computation.

══════════════════════════════════════════════════════════════
COPY THE FOLLOWING CODE SKELETON EXACTLY.
Fill in the <...> placeholders. Do not rename variables or restructure the try/except.
══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

try:
    df = pd.read_csv("{dataset_path}")
    target_column = {target_val}

    # ── Step 1: Shape & basic counts ───────────────────────────────────────
    num_rows, num_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4)

    # ── Step 2: Feature type separation ────────────────────────────────────
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── Step 3: Sample records (first 5 rows — for human inspection only) ──
    sample_records = df.head(5).to_dict(orient="records")

    # ── Step 4: Per-column univariate profile ───────────────────────────────
    # Covers: dtype, missing values, unique count, skewness, kurtosis,
    # outlier count (IQR method), and top value frequencies for categoricals.
    columns_info = []
    for col in df.columns:
        col_info = {{
            "name":          col,
            "dtype":         str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct":   round(float(df[col].isna().mean() * 100), 4),
            "unique_count":  int(df[col].nunique()),
        }}
        if col in numeric_cols:
            _stats = df[col].describe()
            _q1    = float(df[col].quantile(0.25))
            _q3    = float(df[col].quantile(0.75))
            _iqr   = _q3 - _q1
            _lower = _q1 - 1.5 * _iqr
            _upper = _q3 + 1.5 * _iqr
            col_info.update({{
                "mean":          round(float(_stats["mean"]),       4),
                "std":           round(float(_stats["std"]),        4),
                "min":           round(float(_stats["min"]),        4),
                "pct25":         round(_q1,                         4),
                "median":        round(float(df[col].median()),     4),
                "pct75":         round(_q3,                         4),
                "max":           round(float(_stats["max"]),        4),
                "skewness":      round(float(df[col].skew()),       4),
                "kurtosis":      round(float(df[col].kurt()),       4),
                "outlier_count": int(((df[col] < _lower) | (df[col] > _upper)).sum()),
                "outlier_pct":   round(float(((df[col] < _lower) | (df[col] > _upper)).mean() * 100), 4),
            }})
        else:
            _vc = df[col].value_counts()
            col_info.update({{
                "top_values":          _vc.head(10).to_dict(),
                "bottom_values":       _vc.tail(5).to_dict(),
                "is_high_cardinality": bool(df[col].nunique() > 50),
                "mode":                str(_vc.index[0]) if len(_vc) > 0 else None,
            }})
        columns_info.append(col_info)

    # ── Step 5: Correlation matrix (numeric × numeric, upper triangle only) ─
    # Keys are "colA_colB" — self-correlations (diagonal) are excluded.
    if len(numeric_cols) >= 2:
        _cdf = df[numeric_cols].corr().round(4)
        corr_matrix = {{}}
        for _i in range(len(numeric_cols)):
            for _j in range(_i + 1, len(numeric_cols)):   # i < j → no self-pairs
                _k = f"{{numeric_cols[_i]}}_{{numeric_cols[_j]}}"
                corr_matrix[_k] = float(_cdf.iloc[_i, _j])

        # Also identify highly correlated pairs (|r| >= 0.8)
        high_corr_pairs = {{k: v for k, v in corr_matrix.items() if abs(v) >= 0.8}}
    else:
        corr_matrix      = {{}}
        high_corr_pairs  = {{}}

    # ── Step 6: Target column analysis ─────────────────────────────────────
    {target_analysis_block}

    # ── Step 7: Feature-target correlations ────────────────────────────────
    # RULE: dict key must be ONLY the feature name — never "feat_feat" or "feat_target".
    {feat_corr_block}

    # ── Step 8: Skewness summary (numeric columns only) ────────────────────
    skewness_summary = {{
        col: round(float(df[col].skew()), 4)
        for col in numeric_cols
        if df[col].notna().sum() > 1
    }}
    highly_skewed = {{k: v for k, v in skewness_summary.items() if abs(v) > 1.0}}

    # ── Step 9: Missing value summary ──────────────────────────────────────
    missing_summary = {{
        col: {{
            "count": int(df[col].isna().sum()),
            "pct":   round(float(df[col].isna().mean() * 100), 4),
        }}
        for col in df.columns if df[col].isna().any()
    }}
    cols_high_missing = [c for c, v in missing_summary.items() if v["pct"] > 30]

    # ── Step 10: Inject LLM preview context ────────────────────────────────
    dataset_context = {context_literal}

    # ── Assemble summary ────────────────────────────────────────────────────
    summary = {{
        "dataset_path":               "{dataset_path}",
        "task_type":                  "{task_type}",
        "target_column":              target_column,
        "num_rows":                   num_rows,
        "num_cols":                   num_cols,
        "memory_usage_mb":            memory_usage_mb,
        "duplicate_rows":             duplicate_rows,
        "numeric_columns":            numeric_cols,
        "categorical_columns":        categorical_cols,
        "sample_records":             sample_records,
        "columns":                    columns_info,
        "correlation_matrix":         corr_matrix,
        "high_correlation_pairs":     high_corr_pairs,
        "target_analysis":            target_analysis,
        "feature_target_correlations": feature_target_corr,
        "skewness_summary":           skewness_summary,
        "highly_skewed_columns":      highly_skewed,
        "missing_summary":            missing_summary,
        "cols_high_missing":          cols_high_missing,
        "dataset_context":            dataset_context,
    }}
    print(json.dumps(summary, indent=2, default=str))

except Exception as _e:
    import traceback
    print(json.dumps({{"error": str(_e), "traceback": traceback.format_exc()}}, indent=2))

══════════════════════════════════════════════════════════════
END OF SKELETON
══════════════════════════════════════════════════════════════

Output ONLY the Python code. Do not add markdown fences."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    code = response.choices[0].message.content.strip()
    code = _strip_markdown(code)
    logger.info("Data Understanding Agent generated code.")
    return code


def fix_understanding_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("Data Understanding Agent fixing code (attempt %d)...", attempt)

    prompt = f"""The following Python code failed to execute.

ERROR:
{stderr}

STDOUT (may be partial):
{stdout}

CURRENT CODE:
{current_code}

Fix the code so it runs correctly and prints the full JSON summary.
IMPORTANT: keys in feature_target_correlations must be ONLY the feature column name (e.g. "age"),
never a doubled name like "age_age" or "age_target".
The correlation_matrix must use upper-triangle pairs only (i < j) — no self-pairs.
Output ONLY the complete fixed Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    fixed = response.choices[0].message.content.strip()
    fixed = _strip_markdown(fixed)
    logger.info("Data Understanding Agent returned fix.")
    return fixed


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        current_code = current_code_path.read_text()
        return fix_understanding_code(current_code, stderr, stdout, attempt)
    return callback


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(dataset_path: str, task_type: str, target_column: str = None) -> dict:
    # Phase 1: LLM reviews first 1000 rows for business context
    logger.info("Phase 1: Previewing dataset for business context...")
    dataset_context = preview_dataset_context(dataset_path, task_type, target_column)

    # Phase 2: Generate comprehensive EDA code informed by that context
    logger.info("Phase 2: Generating full EDA code...")
    code = generate_understanding_code(dataset_path, task_type, target_column, dataset_context)

    script_path = GENERATED_CODE_DIR / "step1_understanding.py"
    script_path.write_text(code)
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step1_understanding.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
