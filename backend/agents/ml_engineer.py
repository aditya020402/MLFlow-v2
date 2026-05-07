"""ML Engineer Agent — generates step3_ml.py (iteration 0 / baseline model)."""

import json
import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert ML Engineer Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- ALWAYS split data: train_test_split(X, y, test_size=0.2, random_state=42)
- ALWAYS fit preprocessing (scalers, encoders) on TRAIN set only, transform both.
- Metrics must be computed on the TEST set ONLY — never on training data.
- Wrap the final metrics JSON in METRICS_JSON_START / METRICS_JSON_END markers.
- Save the trained model with pickle to the exact path given.
- Include a self-check: reload the saved model, re-predict on test set, log a warning if accuracy differs.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks."""

ALGORITHM_SELECTION_PROMPT = """You are an ML algorithm selection expert.
Respond ONLY with a JSON object — no markdown, no extra text:
{
  "algorithm": "RandomForestClassifier",
  "reason": "2-3 sentence rationale covering: (1) which dataset characteristics made this the best choice, (2) which alternative algorithms you considered and why you rejected them, (3) what performance outcome you expect.",
  "hyperparameters": {"n_estimators": 100, "max_depth": 10}
}"""


def _metrics_block(task_type: str, algo: str) -> str:
    if "classif" in task_type:
        return (
            "accuracy_score, f1_score (weighted), precision_score (weighted), "
            "recall_score (weighted), and classification_report as a string"
        )
    if "regress" in task_type:
        return "r2_score, mean_squared_error, mean_absolute_error, sqrt(mse) as rmse"
    return "silhouette_score (if possible), inertia (if KMeans)"


def _metrics_dict_template(task_type: str, algo: str, model_path: str) -> str:
    base = f"""{{
  "algorithm": "{algo}",
  "iteration": 0,
  "task_type": "{task_type}",
  "model_path": "{model_path}",
  "train_samples": <int>,
  "test_samples": <int>,
  "hyperparameters": <dict of actual hyperparameters passed to {algo}(...)>,"""
    if "classif" in task_type:
        base += '\n  "accuracy": <float>,\n  "f1": <float>,\n  "precision": <float>,\n  "recall": <float>'
    elif "regress" in task_type:
        base += '\n  "r2_score": <float>,\n  "rmse": <float>,\n  "mae": <float>'
    else:
        base += '\n  "silhouette_score": <float>'
    return base + "\n}"


def select_algorithm(
    task_type: str,
    analysis_output: str,
    understanding_output: str,
    target_column: str = None,
    human_feedback: str = "",
) -> dict:
    """Ask GPT-4o to freely pick the best baseline algorithm."""
    feedback_section = f"\nHuman feedback / instructions:\n{human_feedback}" if human_feedback.strip() else ""

    prompt = f"""Select the BEST ML algorithm for this dataset. Reason freely — pick whatever will work best given the data characteristics, do not restrict yourself to any predefined list.

Task type: {task_type}
{"Target column: " + target_column if target_column else ""}

Data Understanding (summary):
{understanding_output[:2000]}

Data Analysis (summary):
{analysis_output[:2000]}
{feedback_section}

Consider: dataset size, feature types, class imbalance, linearity of relationships, interpretability needs, and any human feedback above.

For the `reason` field write 2-3 sentences that clearly explain:
1. Which specific data characteristics (size, feature types, imbalance, etc.) make this algorithm the best fit.
2. Which alternative algorithms you considered and why you rejected each.
3. What performance outcome you expect and why.

Respond ONLY with the JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": ALGORITHM_SELECTION_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse algorithm JSON, using default.")
        if task_type == "supervised_classification":
            return {
                "algorithm": "RandomForestClassifier",
                "reason": "Robust default",
                "hyperparameters": {"n_estimators": 100, "random_state": 42},
            }
        if task_type == "supervised_regression":
            return {
                "algorithm": "RandomForestRegressor",
                "reason": "Robust default",
                "hyperparameters": {"n_estimators": 100, "random_state": 42},
            }
        return {
            "algorithm": "KMeans",
            "reason": "Default clustering",
            "hyperparameters": {"n_clusters": 3, "random_state": 42},
        }


def generate_ml_code(
    dataset_path: str,
    task_type: str,
    target_column: str,
    algorithm_info: dict,
    understanding_output: str,
    analysis_output: str,
    human_feedback: str = "",
) -> str:
    """Generate step3_ml.py (baseline / iteration-0 model)."""
    algo = algorithm_info["algorithm"]
    hyperparams = json.dumps(algorithm_info.get("hyperparameters", {}))
    reason = algorithm_info.get("reason", "")
    model_path = str(OUTPUTS_DIR / "model_iter0.pkl")
    stratify = "stratify=y, " if "classif" in task_type else ""
    scale = "Regressor" in algo or "SV" in algo or "Logistic" in algo or "Ridge" in algo or "Linear" in algo

    feedback_section = f"\nHuman feedback / instructions:\n{human_feedback}" if human_feedback.strip() else ""

    prompt = f"""Write Python code to train a {algo} model (baseline, iteration 0).

Dataset: {dataset_path}
Task type: {task_type}
{"Target column: " + target_column if target_column else "No target (unsupervised)"}
Algorithm: {algo}
Hyperparameters: {hyperparams}
Reason: {reason}
{feedback_section}

Data Understanding:
{understanding_output[:1500]}

Data Analysis:
{analysis_output[:1500]}

Requirements (follow EXACTLY):
1. pandas read_csv to load dataset
2. Preprocessing:
   a. Drop columns where missing > 50%
   b. Fill numeric NaN appropriately (median for skewed, mean for symmetric — match what the analysis step decided)
   c. LabelEncode all object/category columns (fit on train only)
   {"d. Separate X (all except target) and y (target='" + target_column + "')" if target_column else "d. Use all columns as X"}
3. train_test_split(X, y, test_size=0.2, random_state=42, {stratify})
   {"(unsupervised: use all data)" if task_type == "unsupervised" else ""}
4. {"StandardScaler: fit on X_train, transform X_train and X_test" if scale else "No scaling needed"}
5. Train {algo} with: {hyperparams}
6. Compute on TEST SET ONLY: {_metrics_block(task_type, algo)}
7. Save model to: {model_path}  (use pickle.dump)
8. SELF-CHECK:
   loaded_model = pickle.load(open("{model_path}", "rb"))
   self_check_preds = loaded_model.predict(X_test{"_scaled" if scale else ""})
   {"self_check_acc = accuracy_score(y_test, self_check_preds)" if "classif" in task_type else "self_check_r2 = r2_score(y_test, self_check_preds)"}
   if {"abs(self_check_acc - accuracy) > 1e-6" if "classif" in task_type else "abs(self_check_r2 - r2) > 1e-6"}:
       import warnings
       warnings.warn("SELF-CHECK MISMATCH — model reload produced different results!")
9. Print metrics in this EXACT format (no other print before/after the block):

print("METRICS_JSON_START")
print(json.dumps(metrics, indent=2, default=str))
print("METRICS_JSON_END")

Where metrics is:
{_metrics_dict_template(task_type, algo, model_path)}

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    code = response.choices[0].message.content.strip()
    code = _strip_markdown(code)
    logger.info("ML Engineer Agent generated baseline code.")
    return code


def fix_ml_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info("ML Engineer Agent fixing code (attempt %d)...", attempt)

    prompt = f"""The following Python code failed.

ERROR:
{stderr}

STDOUT (partial):
{stdout}

CODE:
{current_code}

Fix it. Output ONLY the complete fixed Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    fixed = response.choices[0].message.content.strip()
    fixed = _strip_markdown(fixed)
    logger.info("ML Engineer Agent returned fix.")
    return fixed


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        current_code = current_code_path.read_text()
        return fix_ml_code(current_code, stderr, stdout, attempt)
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
    target_column: str,
    understanding_output: str,
    analysis_output: str,
    human_feedback: str = "",
) -> dict:
    """Generate baseline ML code, write it, return metadata dict."""
    algorithm_info = select_algorithm(
        task_type, analysis_output, understanding_output, target_column, human_feedback
    )
    logger.info(
        "Selected algorithm: %s — %s",
        algorithm_info["algorithm"],
        algorithm_info.get("reason", ""),
    )

    code = generate_ml_code(
        dataset_path, task_type, target_column,
        algorithm_info, understanding_output, analysis_output, human_feedback,
    )
    script_path = GENERATED_CODE_DIR / "step3_ml.py"
    script_path.write_text(code)
    logger.info("Written: %s", script_path)

    return {
        "script_name": "step3_ml.py",
        "script_path": str(script_path),
        "code": code,
        "algorithm_info": algorithm_info,
        "fix_callback": make_fix_callback(script_path),
    }
