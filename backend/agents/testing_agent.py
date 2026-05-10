"""Testing Agent — generates code to run a trained model on a new test CSV and
returns per-row predictions + aggregate metrics as structured JSON."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR        = Path(__file__).parent.parent / "outputs"

from agents.llm_client import client, MODEL
from agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert ML Testing Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable.
- NEVER retrain or fit any model — only load and predict.
- FEATURE ALIGNMENT IS MANDATORY: After all preprocessing, ALWAYS align X to the saved feature list:
    features_path = model_path.replace(".pkl", "_features.pkl")
    if os.path.exists(features_path):
        saved_cols = pickle.load(open(features_path, "rb"))
        for col in saved_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[saved_cols]
  If features_path does not exist, fall back to model.n_features_in_ to validate column count.
- ALWAYS include ALL necessary imports at the top of the script. In particular:
    import json, pickle, os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
  Include any other sklearn or scipy imports that the preprocessing steps require.
- For CLUSTERING models (KMeans, DBSCAN, AgglomerativeClustering, etc.): ALWAYS use
  model.fit_predict(X) — NEVER model.predict(X). Many clustering models have no predict().
- NEVER use mean_squared_error(..., squared=False) — removed in scikit-learn 1.4.
  Compute RMSE as: rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
- The script MUST use this EXACT try/except structure — the markers must appear even on error:

    try:
        # ... all code ...
        results = { ... }
        print("TEST_RESULTS_START")
        print(json.dumps(results, indent=2, default=str))
        print("TEST_RESULTS_END")
    except Exception as e:
        import sys
        results = {"error": str(e), "task_type": "<task_type>", "predictions": [], "metrics": {}, "summary": {}}
        print("TEST_RESULTS_START")
        print(json.dumps(results, indent=2, default=str))
        print("TEST_RESULTS_END")
        sys.exit(1)
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
ALWAYS ensure all necessary imports are present at the top of the script, including:
    import json, pickle, os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
NEVER use mean_squared_error(..., squared=False) — use float(np.sqrt(mean_squared_error(...))) instead.
For clustering models use fit_predict(X) not predict(X).
The try/except MUST always print TEST_RESULTS_START / TEST_RESULTS_END even on error.
FEATURE MISMATCH FIX: If the error mentions features/shape mismatch, add this block AFTER all preprocessing and BEFORE predict():
    features_path = model_path.replace(".pkl", "_features.pkl")
    if os.path.exists(features_path):
        saved_cols = pickle.load(open(features_path, "rb"))
        for col in saved_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[saved_cols]"""


def _metrics_code(task_type: str, target_col: str) -> str:
    if "classif" in task_type:
        return f"""\
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
_y_pred_s   = pd.Series(y_pred).reset_index(drop=True)
_y_actual_s = pd.Series(y_actual).reset_index(drop=True)
if pd.api.types.is_numeric_dtype(_y_pred_s) and not pd.api.types.is_numeric_dtype(_y_actual_s):
    from sklearn.preprocessing import LabelEncoder as _LE
    _le = _LE()
    _le.fit(sorted(_y_actual_s.unique()))
    y_actual_cmp = pd.Series(_le.transform(_y_actual_s)).astype(str)
    y_pred_cmp   = _y_pred_s.astype(str)
else:
    y_pred_cmp   = _y_pred_s.astype(str)
    y_actual_cmp = _y_actual_s.astype(str)
metrics = {{
    "accuracy":  round(float(accuracy_score(y_actual_cmp, y_pred_cmp)), 4),
    "f1":        round(float(f1_score(y_actual_cmp, y_pred_cmp, average="weighted", zero_division=0)), 4),
    "precision": round(float(precision_score(y_actual_cmp, y_pred_cmp, average="weighted", zero_division=0)), 4),
    "recall":    round(float(recall_score(y_actual_cmp, y_pred_cmp, average="weighted", zero_division=0)), 4),
    "classification_report": classification_report(y_actual_cmp, y_pred_cmp, zero_division=0),
}}
total   = len(y_pred_cmp)
correct = int((y_pred_cmp == y_actual_cmp).sum())
wrong   = total - correct
summary = {{"total": total, "correct": correct, "wrong": wrong, "accuracy_pct": round(correct / total * 100, 2)}}"""
    if "regress" in task_type:
        return f"""\
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
y_pred_f   = pd.Series(y_pred).reset_index(drop=True).astype(float)
y_actual_f = pd.Series(y_actual).reset_index(drop=True).astype(float)
mse = float(mean_squared_error(y_actual_f, y_pred_f))
metrics = {{
    "r2_score": round(float(r2_score(y_actual_f, y_pred_f)), 4),
    "rmse":     round(float(np.sqrt(mse)), 4),
    "mae":      round(float(mean_absolute_error(y_actual_f, y_pred_f)), 4),
    "mse":      round(mse, 4),
}}
summary = {{"total": len(y_pred_f)}}"""
    # Clustering — compute silhouette if possible
    return """\
try:
    from sklearn.metrics import silhouette_score as _sil
    _labels = pd.Series(y_pred).reset_index(drop=True)
    _sil_val = round(float(_sil(X, _labels)), 4) if len(set(_labels)) > 1 else -1.0
except Exception:
    _sil_val = -1.0
metrics = {"silhouette_score": _sil_val}
summary = {"total": len(y_pred)}"""


def generate_test_code(
    test_csv_path: str,
    model_path: str,
    task_type: str,
    target_column: str | None,
    training_code: str,
    session_id: str = "",
) -> str:
    is_cluster = not ("classif" in task_type or "regress" in task_type)
    has_target = bool(target_column) and not is_cluster
    metrics_block = _metrics_code(task_type, target_column or "")

    prompt = f"""Write Python code to run the already-trained model on a new test CSV and produce per-row predictions.

Model path:    {model_path}
Test CSV path: {test_csv_path}
Task type:     {task_type}
{"Target column: " + target_column if target_column else "No target (unsupervised — predict cluster labels)"}

=== TRAINING CODE (extract preprocessing steps from here — do NOT retrain) ===
{training_code[:6000]}

=== REQUIREMENTS ===

0. Start with these imports (add others as needed by the training code):
   import json, pickle, os
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler

1. Load the model:
   model = pickle.load(open("{model_path}", "rb"))

2. Load the test CSV:
   import pandas as pd
   df_test = pd.read_csv("{test_csv_path}")

3. PREPROCESSING — apply the EXACT same steps used in the training code above:
   - Drop the same columns that were dropped in training (constant, text, high-cardinality if dropped)
   - Apply the same missing-value strategy (median/mean/mode per column)
   - Apply the same encoding (LabelEncoder / get_dummies / frequency map) — but DO NOT fit on test data;
     reconstruct the mapping using the training code's logic applied to the test set only
   - Apply scaling if the training code used StandardScaler — fit a NEW scaler on test set
     (since we don't have the saved scaler — this is acceptable for inference evaluation)
   {"- Separate X and y_actual: X = df_test.drop(columns=['" + target_column + "']) if '" + target_column + "' in df_test.columns else df_test.copy(); y_actual_exists = '" + target_column + "' in df_test.columns" if has_target else "- X = df_test (all columns)"}

   MANDATORY FEATURE ALIGNMENT (add this block after all preprocessing, before predict):
   import os
   model_path = "{model_path}"
   features_path = model_path.replace(".pkl", "_features.pkl")
   if os.path.exists(features_path):
       saved_cols = pickle.load(open(features_path, "rb"))
       for col in saved_cols:
           if col not in X.columns:
               X[col] = 0
       X = X[saved_cols]
   This ensures X has EXACTLY the same columns (count and order) as training, preventing feature mismatch errors.

4. Predict:
   {"y_pred = model.fit_predict(X)  # clustering: ALWAYS use fit_predict — DBSCAN/AgglomerativeClustering have no predict()" if is_cluster else "y_pred = model.predict(X)"}
   y_pred = pd.Series(y_pred).reset_index(drop=True)

5. {"Compute metrics (ONLY if target column present in test CSV):" if has_target else "No supervised metrics — clustering only."}
   {"y_actual = df_test['" + target_column + "'] if y_actual_exists else None" if has_target else ""}
   {"has_actual = y_actual_exists" if has_target else "has_actual = False"}
   if has_actual:
       y_actual = y_actual.reset_index(drop=True)
       {metrics_block}
   else:
       metrics = {{}}
       summary = {{"total": len(y_pred)}}

6. Build per-row records (use X with its original column values for display):
   records = []
   X_display = X.reset_index(drop=True)
   feature_cols = list(X_display.columns)
   for i in range(len(X_display)):
       row = {{col: X_display.iloc[i][col] for col in feature_cols[:20]}}  # cap at 20 feature cols
       {"row['actual']    = str(y_actual.iloc[i]) if has_actual else None" if has_target else ""}
       {"row['predicted'] = str(y_pred_cmp.iloc[i]) if has_actual else str(y_pred.iloc[i])" if "classif" in task_type else "row['predicted'] = str(y_pred.iloc[i])"}
       {"row['correct']   = bool(y_pred_cmp.iloc[i] == y_actual_cmp.iloc[i]) if has_actual else None" if "classif" in task_type else ""}
       {"row['error']     = round(float(y_actual_f.iloc[i]) - float(y_pred_f.iloc[i]), 4) if has_actual else None" if "regress" in task_type else ""}
       {"row['cluster']   = int(y_pred.iloc[i])" if is_cluster else ""}
       records.append({{k: v for k, v in row.items() if v is not None}})

7. Wrap output in EXACTLY this try/except so markers always appear:
   try:
       results = {{
           "task_type":        "{task_type}",
           "target_column":    {"'" + target_column + "'" if target_column else "None"},
           "has_actual_labels": has_actual,
           "predictions":      records,
           "metrics":          metrics,
           "summary":          summary,
           {"'" + '"cluster_counts": {str(int(k)): int(v) for k, v in pd.Series(y_pred).value_counts().items()},' + "'" if is_cluster else ""}
       }}
       print("TEST_RESULTS_START")
       print(json.dumps(results, indent=2, default=str))
       print("TEST_RESULTS_END")
   except Exception as _e:
       import sys
       print("TEST_RESULTS_START")
       print(json.dumps({{"error": str(_e), "task_type": "{task_type}", "predictions": [], "metrics": {{}}, "summary": {{}}}}, indent=2))
       print("TEST_RESULTS_END")
       sys.exit(1)

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
    )
    if response.usage:
        _tok("testing_agent", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Testing Agent generated code for session %s.", session_id)
    return code


def fix_test_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("Testing Agent fixing code (attempt %d)...", attempt)
    prompt = f"""Code failed.

ERROR:
{stderr}

STDOUT:
{stdout}

CODE:
{current_code}

Fix it. Output ONLY the complete fixed Python code."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    if response.usage:
        _tok("testing_agent", response.usage.prompt_tokens, response.usage.completion_tokens)
    return _strip_markdown(response.choices[0].message.content.strip())


def parse_test_results(stdout: str) -> dict:
    start = stdout.find("TEST_RESULTS_START")
    end   = stdout.find("TEST_RESULTS_END")
    if start != -1 and end != -1 and end > start:
        raw = stdout[start + len("TEST_RESULTS_START"):end].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(
    test_csv_path: str,
    model_path: str,
    task_type: str,
    target_column: str | None,
    training_code: str,
    session_id: str = "",
) -> dict:
    """Generate, execute, and return test results."""
    from tools.runner import run_script_with_retry

    code = generate_test_code(
        test_csv_path, model_path, task_type,
        target_column, training_code, session_id,
    )

    script_name = f"step_test_{session_id or 'default'}.py"
    script_path = GENERATED_CODE_DIR / script_name
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)

    def _fix_callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_test_code(script_path.read_text(), stderr, stdout, attempt)

    result = run_script_with_retry(
        script_name,
        fix_callback=_fix_callback,
        max_retries=5,
        timeout=120,
    )

    if not result.success:
        return {
            "error": "Test script failed after retries.",
            "stderr": result.stderr[:2000],
        }

    data = parse_test_results(result.stdout)
    if not data:
        return {
            "error": "Could not parse test results from script output.",
            "stdout": result.stdout[:2000],
        }

    return data
