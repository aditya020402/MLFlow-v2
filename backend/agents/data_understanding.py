import logging
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

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
IMPORTANT: keys in feature_target_correlations must be ONLY the feature column name (e.g. "age"),
never a doubled name like "age_age"."""


def generate_understanding_code(dataset_path: str, task_type: str, target_column: str = None) -> str:
    target_val = f'"{target_column}"' if target_column else "None"
    is_classif = "classif" in task_type
    is_regress  = "regress" in task_type

    # Build the feature-target correlation block as explicit code so the LLM
    # cannot get the dict key format wrong.
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

    prompt = f"""Write Python code to perform comprehensive data understanding on the dataset at:
  {dataset_path}

Task type: {task_type}
Target column: {target_column if target_column else "None (unsupervised)"}

Analyse the ENTIRE dataset — never call df.sample().

══════════════════════════════════════════
COPY THE FOLLOWING CODE SKELETON EXACTLY.
Fill in the <...> placeholders. Do not rename variables.
══════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

try:
    df = pd.read_csv("{dataset_path}")
    target_column = {target_val}

    # ── shape & duplicates ──────────────────────────────────────
    num_rows, num_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    # ── column type lists ───────────────────────────────────────
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ── per-column profile ──────────────────────────────────────
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
                "outlier_count": int(((df[col] < _q1 - 1.5*_iqr) | (df[col] > _q3 + 1.5*_iqr)).sum()),
            }})
        else:
            col_info.update({{
                "top_values":          df[col].value_counts().head(10).to_dict(),
                "is_high_cardinality": bool(df[col].nunique() > 50),
            }})
        columns_info.append(col_info)

    # ── correlation matrix (numeric × numeric) ──────────────────
    # Each key is a column name; its value is a dict of {{other_col: pearson_r}}.
    # This captures CROSS-feature correlations, not self-correlations.
    if len(numeric_cols) >= 2:
        _cdf = df[numeric_cols].corr().round(4)
        corr_matrix = {{
            col: {{other: float(v) for other, v in row.items()}}
            for col, row in _cdf.to_dict().items()
        }}
    else:
        corr_matrix = {{}}

    # ── target column analysis ──────────────────────────────────
    target_analysis = {{}}
    if target_column is not None and target_column in df.columns:
        _tdtype = str(df[target_column].dtype)
        if "{task_type}" and "classif" in "{task_type}":
            _vc = df[target_column].value_counts()
            target_analysis = {{
                "dtype":            _tdtype,
                "class_counts":     {{str(k): int(v) for k, v in _vc.items()}},
                "class_balance":    {{str(k): round(int(v) / num_rows * 100, 4) for k, v in _vc.items()}},
                "imbalance_ratio":  round(float(_vc.iloc[0]) / float(_vc.iloc[-1]), 4),
            }}
        else:
            _ts = df[target_column].describe()
            target_analysis = {{
                "dtype":           _tdtype,
                "target_stats":    {{k: round(float(v), 4) for k, v in _ts.items()}},
                "target_skewness": round(float(df[target_column].skew()), 4),
            }}

    # ── feature-target correlations ─────────────────────────────
    # RULE: dict key must be ONLY the feature name — never "feat_feat" or "feat_target".
    # Copy the block below verbatim:
    {feat_corr_block}

    # ── assemble summary ────────────────────────────────────────
    summary = {{
        "dataset_path":               "{dataset_path}",
        "task_type":                  "{task_type}",
        "target_column":              target_column,
        "num_rows":                   num_rows,
        "num_cols":                   num_cols,
        "duplicate_rows":             duplicate_rows,
        "numeric_columns":            numeric_cols,
        "categorical_columns":        categorical_cols,
        "columns":                    columns_info,
        "correlation_matrix":         corr_matrix,
        "target_analysis":            target_analysis,
        "feature_target_correlations": feature_target_corr,
    }}
    print(json.dumps(summary, indent=2, default=str))

except Exception as _e:
    import traceback
    print(json.dumps({{"error": str(_e), "traceback": traceback.format_exc()}}, indent=2))

══════════════════════════════════════════
END OF SKELETON
══════════════════════════════════════════

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
    code = generate_understanding_code(dataset_path, task_type, target_column)
    script_path = GENERATED_CODE_DIR / "step1_understanding.py"
    script_path.write_text(code)
    logger.info("Written: %s", script_path)
    return {
        "script_name": "step1_understanding.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
