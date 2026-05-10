"""ML Engineer Agent — generates step3_ml.py (iteration 0 / baseline model)."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

from agents.llm_client import client, MODEL
from agents.token_tracker import record as _tok

SYSTEM_PROMPT = """You are an expert ML Engineer Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- For SUPERVISED tasks (classification/regression): split with train_test_split(X, y, test_size=0.2, random_state=42).
- For CLUSTERING (unsupervised) tasks:
    * Split X only: X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  — NO y whatsoever.
    * NEVER invent or guess a target column. NEVER call train_test_split with a y argument.
    * Fit the model on X_train only: model.fit(X_train) or model.fit(X_train_scaled).
    * Evaluate with fit_predict on X_test: predictions = model.fit_predict(X_test).
    * Compute silhouette_score(X_test, predictions) — only if len(set(predictions)) > 1.
    * NEVER use RandomizedSearchCV for clustering — it requires a y and a scorer that don't apply.
- ALWAYS fit preprocessing (scalers, encoders) on TRAIN set only, transform both.
- Metrics must be computed on the TEST set ONLY — never on training data.
- Wrap the final metrics JSON in METRICS_JSON_START / METRICS_JSON_END markers.
- Save the trained model with pickle to the exact path given.
- Include a self-check: reload the saved model, re-predict on test set, log a warning if results differ.
- NEVER use mean_squared_error(..., squared=False) — the `squared` parameter was removed in scikit-learn 1.4.
  Always compute RMSE as: import numpy as np; rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
NEVER use mean_squared_error(..., squared=False) — removed in scikit-learn 1.4.
Compute RMSE as: rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
CRITICAL: If a target column is provided in the context, use that EXACT column name. NEVER hardcode 'target' as a column name. NEVER fall back to the last column — if the column is missing, raise a clear error naming the expected column."""

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
        return "r2_score, mean_squared_error, mean_absolute_error — compute rmse as float(np.sqrt(mean_squared_error(y_test, predictions))) — NEVER use squared=False"
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


def _build_data_context(analysis_data: dict, understanding_output: str) -> str:
    """Format analysis_data into a concise data-characteristics summary for algorithm selection."""
    if not analysis_data:
        return understanding_output[:2000]

    lines = []

    shape = analysis_data.get("original_shape") or analysis_data.get("cleaned_shape")
    if shape:
        lines.append(f"Dataset size: {shape[0]} rows × {shape[1]} columns")

    cleaned = analysis_data.get("cleaned_shape")
    if cleaned:
        lines.append(f"After cleaning: {cleaned[0]} rows × {cleaned[1]} columns")

    col_types = analysis_data.get("col_types", {})
    if col_types:
        by_type: dict[str, list] = {}
        for col, ctype in col_types.items():
            by_type.setdefault(ctype, []).append(col)
        for ctype, cols in by_type.items():
            lines.append(f"{ctype} columns ({len(cols)}): {', '.join(cols[:8])}{'…' if len(cols) > 8 else ''}")

    hcp = analysis_data.get("high_correlation_pairs", [])
    if hcp:
        hcp_strs = [item["pair"] if isinstance(item, dict) else str(item) for item in hcp[:6]]
        lines.append(f"High-correlation pairs (|r|>0.90): {', '.join(hcp_strs)}")

    top = analysis_data.get("top_features", [])
    if top:
        lines.append(f"Top predictive features: {', '.join(top[:8])}")

    imb_ratio = analysis_data.get("imbalance_ratio", 1.0)
    imb_rec = analysis_data.get("imbalance_recommendation", "")
    if imb_rec and imb_rec != "n/a":
        dist = analysis_data.get("class_distribution", {})
        dist_str = ", ".join(f"{k}: {v}" for k, v in list(dist.items())[:5])
        lines.append(f"Class imbalance ratio: {imb_ratio:.1f}x — {{{dist_str}}}")
        lines.append(f"Imbalance recommendation: {imb_rec}")

    outlier = analysis_data.get("outlier_summary", {})
    heavy = [col for col, v in outlier.items() if "high outlier rate" in v.get("action", "")]
    if heavy:
        lines.append(f"Columns with heavy outliers (not capped): {', '.join(heavy)}")

    insights = analysis_data.get("feature_insights", [])
    for insight in insights[:4]:
        lines.append(f"• {insight}")

    plot_insights = analysis_data.get("plot_insights", [])
    if plot_insights:
        lines.append("Plot-derived insights:")
        for pi in plot_insights[:5]:
            lines.append(f"  • {pi}")

    ml_recs = analysis_data.get("ml_model_recommendations", {})
    if ml_recs:
        algo_hints = ml_recs.get("algorithm_hints", [])
        if algo_hints:
            lines.append("Algorithm hints (from data analyst):")
            for h in algo_hints[:4]:
                lines.append(f"  → {h}")
        challenges = ml_recs.get("expected_challenges", [])
        if challenges:
            lines.append("Expected modeling challenges:")
            for c in challenges[:4]:
                lines.append(f"  ⚠ {c}")
        preproc = ml_recs.get("preprocessing_notes", [])
        if preproc:
            lines.append("Remaining preprocessing for ML engineer:")
            for p in preproc[:4]:
                lines.append(f"  - {p}")
        feat_candidates = ml_recs.get("feature_importance_candidates", [])
        if feat_candidates:
            lines.append(f"Feature importance candidates: {', '.join(feat_candidates[:10])}")

    return "\n".join(lines)


def select_algorithm(
    task_type: str,
    analysis_output: str,
    understanding_output: str,
    target_column: str = None,
    human_feedback: str = "",
    tried_algorithms: list = None,
    analysis_data: dict = None,
) -> dict:
    """Ask GPT-4o to pick the best baseline algorithm, excluding already-tried ones."""
    tried_algorithms = tried_algorithms or []
    is_cluster = not ("classif" in task_type or "regress" in task_type)
    feedback_section = f"\nHuman feedback / instructions:\n{human_feedback}" if human_feedback.strip() else ""
    exclusion_line = (
        f"\nDO NOT pick any of these — already tried: {tried_algorithms}"
        if tried_algorithms else ""
    )

    data_context = _build_data_context(analysis_data or {}, understanding_output)

    _CLUSTERING_ALGOS = {
        "KMeans", "MiniBatchKMeans", "AgglomerativeClustering",
        "GaussianMixture", "DBSCAN", "Birch", "SpectralClustering",
        "AffinityPropagation", "MeanShift", "OPTICS",
    }

    prompt = f"""Select the BEST ML algorithm for this dataset. Reason from the actual data characteristics and analyst recommendations below — do not default to popular choices.

Task type: {task_type}
{"Target column: " + target_column if target_column else "No target (unsupervised)"}
{exclusion_line}

=== DATA CHARACTERISTICS & ANALYST RECOMMENDATIONS ===
(Includes plot-derived insights, algorithm hints, expected challenges, and preprocessing notes
 from the data analyst — these are grounded in the actual dataset statistics and visualisations.)
{data_context}

=== DATA UNDERSTANDING (narrative) ===
{understanding_output[:1000]}
{feedback_section}

Think carefully about ALL of the above, especially:
- Algorithm hints and expected challenges flagged by the data analyst
- Dataset size: small datasets tolerate complex models; large datasets benefit from simpler/faster ones
- Feature types: all-numeric vs mixed vs high-cardinality categorical
- Class imbalance: if severe, prefer algorithms with native class_weight support
- Correlation structure: highly correlated features favour regularised models
- Outlier presence: tree-based models are more robust; distance-based are sensitive
- Task type: {task_type}
{"- CLUSTERING TASK: you MUST pick a clustering algorithm. Valid choices: KMeans, AgglomerativeClustering, GaussianMixture, DBSCAN, MiniBatchKMeans, Birch, SpectralClustering. DO NOT pick any supervised algorithm (no RandomForest, XGBoost, SVM, LogisticRegression, etc.)." if is_cluster else "- SUPERVISED TASK: do NOT pick a clustering algorithm."}

For the `reason` field write 2-3 sentences that clearly explain:
1. Which specific data characteristics and analyst findings make this algorithm the best fit.
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

    if response.usage:
        _tok("ml_engineer", response.usage.prompt_tokens, response.usage.completion_tokens)
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Fallback defaults per task type (ordered — pick first untried)
    _defaults_classif = [
        ("RandomForestClassifier",     {"n_estimators": 100, "random_state": 42}),
        ("GradientBoostingClassifier", {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42}),
        ("LogisticRegression",         {"C": 1.0, "max_iter": 1000, "random_state": 42}),
        ("XGBClassifier",              {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}),
    ]
    _defaults_regress = [
        ("RandomForestRegressor",     {"n_estimators": 100, "random_state": 42}),
        ("GradientBoostingRegressor", {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42}),
        ("Ridge",                      {"alpha": 1.0}),
        ("XGBRegressor",              {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}),
    ]

    def _pick_default(defaults):
        for name, params in defaults:
            if name not in tried_algorithms:
                return {"algorithm": name, "reason": f"Default fallback — {name} selected as first untried algorithm.", "hyperparameters": params}
        name, params = defaults[0]
        return {"algorithm": name, "reason": "All defaults exhausted; reusing first.", "hyperparameters": params}

    _defaults_cluster = [
        ("KMeans",                  {"n_clusters": 3, "random_state": 42}),
        ("AgglomerativeClustering", {"n_clusters": 3}),
        ("GaussianMixture",         {"n_components": 3, "random_state": 42}),
        ("MiniBatchKMeans",         {"n_clusters": 3, "random_state": 42}),
    ]

    def _pick_cluster_default():
        for name, params in _defaults_cluster:
            if name not in tried_algorithms:
                return {"algorithm": name, "reason": f"Default clustering fallback — {name} selected.", "hyperparameters": params}
        name, params = _defaults_cluster[0]
        return {"algorithm": name, "reason": "All clustering defaults exhausted; reusing first.", "hyperparameters": params}

    try:
        result = json.loads(text)
        chosen = result.get("algorithm", "")

        # Hard guard: clustering task must use a clustering algorithm
        if is_cluster and chosen not in _CLUSTERING_ALGOS:
            logger.warning(
                "LLM picked supervised algorithm '%s' for clustering task — substituting.", chosen
            )
            return _pick_cluster_default()

        # Hard guard: supervised tasks must not use clustering algorithms
        if not is_cluster and chosen in _CLUSTERING_ALGOS:
            logger.warning(
                "LLM picked clustering algorithm '%s' for supervised task — substituting.", chosen
            )
            if "classif" in task_type:
                return _pick_default(_defaults_classif)
            return _pick_default(_defaults_regress)

        # Enforce exclusion list — LLM may ignore it
        if tried_algorithms and chosen in tried_algorithms:
            logger.warning("LLM picked already-tried baseline '%s'; substituting.", chosen)
            if "classif" in task_type:
                return _pick_default(_defaults_classif)
            if "regress" in task_type:
                return _pick_default(_defaults_regress)
            return _pick_cluster_default()

        return result
    except json.JSONDecodeError:
        logger.warning("Could not parse algorithm JSON, using default.")
        if "classif" in task_type:
            return _pick_default(_defaults_classif)
        if "regress" in task_type:
            return _pick_default(_defaults_regress)
        return _pick_cluster_default()


def generate_ml_code(
    dataset_path: str,
    task_type: str,
    target_column: str,
    algorithm_info: dict,
    understanding_output: str,
    analysis_output: str,
    human_feedback: str = "",
    session_id: str = "",
    analysis_data: dict = None,
) -> str:
    """Generate step3_ml.py (baseline / iteration-0 model)."""
    algo = algorithm_info["algorithm"]
    hyperparams = json.dumps(algorithm_info.get("hyperparameters", {}))
    reason = algorithm_info.get("reason", "")
    session_out = OUTPUTS_DIR / (session_id if session_id else "default")
    session_out.mkdir(parents=True, exist_ok=True)
    model_path = str(session_out / "model_iter0.pkl")
    is_cluster = not ("classif" in task_type or "regress" in task_type)
    stratify = "stratify=y, " if "classif" in task_type else ""

    feedback_section = f"\nHuman feedback / instructions:\n{human_feedback}" if human_feedback.strip() else ""

    # Build preprocessing intelligence from structured analysis data
    preproc_intel = _build_preprocessing_intelligence(analysis_data or {}, algo, task_type)

    prompt = f"""Write Python code to train a {algo} model (baseline, iteration 0).

Dataset: {dataset_path}
Task type: {task_type}
{"Target column: " + target_column if target_column else "No target (unsupervised)"}
Algorithm: {algo}
Hyperparameters: {hyperparams}
Reason: {reason}
{feedback_section}

=== PREPROCESSING INTELLIGENCE (from data analysis — follow these decisions) ===
{preproc_intel}

=== REQUIREMENTS (follow EXACTLY) ===
1. pandas read_csv to load dataset

2. SMART PREPROCESSING (based on the intelligence section above):
   a. Drop columns where missing > 50%
   b. Handle remaining missing values per column type — use median/mean for numeric, mode for categorical
   c. ENCODING — use the encoding_recommendations from the intelligence section:
      - "label": LabelEncoder (fit on train only)
      - "onehot": pd.get_dummies on train, reindex test to match train columns
      - "frequency": map each category to its frequency in the TRAIN set only
      - "target": map each category to the mean target value in the TRAIN set only
      - "drop": drop the column entirely
   d. FEATURE SELECTION — drop features with near-zero variance (std < 0.001) after encoding.
      Additionally drop one column from each high-correlation pair listed above (keep the one
      with higher correlation to the target, or keep the first if unsupervised).
   e. CLASS IMBALANCE — if imbalance_recommendation says to use SMOTE or class_weight,
      apply the most appropriate strategy for {algo}:
      - If {algo} supports class_weight: pass class_weight='balanced'
      - Otherwise apply SMOTE from imblearn BEFORE train_test_split (fit on train only)
   {"f. Separate X and y (target='" + target_column + "')" if target_column else "f. X = all columns — this is unsupervised, there is NO target column, do NOT create one"}

{"3. CLUSTERING SPLIT — NO y:" if is_cluster else "3. SUPERVISED SPLIT:"}
   {"X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # NO y argument" if is_cluster else f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, {stratify})"}

4. SCALING — only if the intelligence section says scaling is needed for {algo}:
   StandardScaler fit on X_train, transform X_train and X_test.
   Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting,
   Extra Trees, Decision Tree) do NOT need scaling.

{"5. CLUSTERING FIT + EVALUATE (no RandomizedSearchCV):" if is_cluster else f"5. Train {algo} with: {hyperparams}"}
   {f"""model = {algo}(**{hyperparams})
   model.fit(X_train_scaled if scaling was applied else X_train)
   predictions = model.fit_predict(X_test_scaled if scaling was applied else X_test)
   from sklearn.metrics import silhouette_score
   sil = silhouette_score(X_test_final, predictions) if len(set(predictions)) > 1 else -1.0""" if is_cluster else ""}

6. Compute on TEST SET ONLY: {_metrics_block(task_type, algo)}
   {"For clustering: predictions = model.fit_predict(X_test_final); silhouette_score only if len(set(predictions)) > 1" if is_cluster else ""}

7. Save model AND feature columns:
   import pickle, os
   pickle.dump(model, open("{model_path}", "wb"))
   features_path = "{model_path}".replace(".pkl", "_features.pkl")
   pickle.dump(list(X_train.columns), open(features_path, "wb"))

8. SELF-CHECK — reload model and re-predict; warn if results differ:
   sc_model = pickle.load(open("{model_path}", "rb"))
   {"sc_preds = sc_model.fit_predict(X_test_final)  # clustering: use fit_predict, not predict" if is_cluster else "sc_preds = sc_model.predict(X_test_final)"}
   if not all(sc_preds == predictions):
       import warnings; warnings.warn("SELF-CHECK MISMATCH")

9. Print metrics in this EXACT format:

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

    if response.usage:
        _tok("ml_engineer", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = response.choices[0].message.content.strip()
    code = _strip_markdown(code)
    logger.info("ML Engineer Agent generated baseline code.")
    return code


def fix_ml_code(
    current_code: str,
    stderr: str,
    stdout: str,
    attempt: int,
    dataset_path: str = "",
    task_type: str = "",
    target_column: str = "",
) -> str:
    logger.info("ML Engineer Agent fixing code (attempt %d)...", attempt)

    context_lines = []
    if dataset_path:
        context_lines.append(f"Dataset path: {dataset_path}")
    if task_type:
        context_lines.append(f"Task type: {task_type}")
    if target_column:
        context_lines.append(
            f"Target column: '{target_column}' — use this exact column name, do NOT hardcode 'target' or guess the last column."
        )
    context_block = "\n".join(context_lines)

    prompt = f"""The following Python code failed.

{context_block}

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

    if response.usage:
        _tok("ml_engineer", response.usage.prompt_tokens, response.usage.completion_tokens)
    fixed = response.choices[0].message.content.strip()
    fixed = _strip_markdown(fixed)
    logger.info("ML Engineer Agent returned fix.")
    return fixed


def make_fix_callback(
    current_code_path: Path,
    dataset_path: str = "",
    task_type: str = "",
    target_column: str = "",
):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        current_code = current_code_path.read_text()
        return fix_ml_code(
            current_code, stderr, stdout, attempt,
            dataset_path=dataset_path,
            task_type=task_type,
            target_column=target_column,
        )
    return callback


def _build_preprocessing_intelligence(analysis_data: dict, algo: str, task_type: str) -> str:
    """Format structured analysis_data into an LLM-readable preprocessing context."""
    if not analysis_data:
        return "No structured analysis data available — apply standard preprocessing heuristics."

    sections = []

    # Column type summary
    col_types = analysis_data.get("col_types", {})
    if col_types:
        by_type: dict[str, list] = {}
        for col, ctype in col_types.items():
            by_type.setdefault(ctype, []).append(col)
        type_lines = "\n".join(f"  {ctype}: {', '.join(cols)}" for ctype, cols in by_type.items())
        sections.append(f"COLUMN TYPES:\n{type_lines}")

    # Encoding recommendations
    enc = analysis_data.get("encoding_recommendations", {})
    if enc:
        enc_lines = "\n".join(f"  {col}: {how}" for col, how in enc.items())
        sections.append(f"ENCODING RECOMMENDATIONS (apply exactly in ML code):\n{enc_lines}")

    # Let the LLM decide scaling based on the algorithm's mathematical properties
    sections.append(
        f"SCALING DECISION: The algorithm is {algo}. "
        f"Decide whether StandardScaler is needed based on {algo}'s mathematical properties. "
        f"Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees, Decision Tree, AdaBoost) "
        f"do NOT need scaling. Distance-based models (KNN, SVM, SVR, DBSCAN, KMeans, AgglomerativeClustering), "
        f"linear models (Logistic Regression, Ridge, Lasso, ElasticNet), and neural networks DO need scaling. "
        f"GaussianMixture needs scaling. Apply StandardScaler if needed."
    )

    # High-correlation pairs (feature selection)
    hcp = analysis_data.get("high_correlation_pairs", [])
    if hcp:
        hcp_strs = [item["pair"] if isinstance(item, dict) else str(item) for item in hcp]
        sections.append(f"HIGH-CORRELATION PAIRS (drop one from each pair after encoding):\n  {', '.join(hcp_strs)}")
    else:
        sections.append("HIGH-CORRELATION PAIRS: none detected above 0.90 threshold")

    # Outlier handling (data analyst already winsorized; note it for context)
    outlier = analysis_data.get("outlier_summary", {})
    capped = [(col, v.get("action", "")) for col, v in outlier.items() if "capped" in v.get("action", "")]
    if capped:
        cap_lines = "\n".join(f"  {col}: {action}" for col, action in capped)
        sections.append(f"OUTLIER HANDLING (already applied to cleaned CSV — do NOT re-apply):\n{cap_lines}")

    # Class imbalance
    imb_rec = analysis_data.get("imbalance_recommendation", "")
    if imb_rec and imb_rec != "n/a":
        ratio = analysis_data.get("imbalance_ratio", 1.0)
        dist = analysis_data.get("class_distribution", {})
        dist_str = ", ".join(f"{k}: {v}" for k, v in list(dist.items())[:6])
        sections.append(
            f"CLASS IMBALANCE:\n"
            f"  Imbalance ratio: {ratio:.1f}x\n"
            f"  Class distribution: {{{dist_str}}}\n"
            f"  Recommendation: {imb_rec}"
        )
    else:
        sections.append("CLASS IMBALANCE: balanced — no special handling needed")

    # Feature insights / top features
    top = analysis_data.get("top_features", [])
    if top:
        sections.append(f"TOP PREDICTIVE FEATURES (from correlation/variance analysis):\n  {', '.join(top[:10])}")

    # Plot-derived insights from data analyst
    plot_insights = analysis_data.get("plot_insights", [])
    if plot_insights:
        pi_lines = "\n".join(f"  • {pi}" for pi in plot_insights[:5])
        sections.append(f"PLOT-DERIVED INSIGHTS (data analyst observations):\n{pi_lines}")

    # Structured ML recommendations from data analyst
    ml_recs = analysis_data.get("ml_model_recommendations", {})
    if ml_recs:
        rec_lines = []
        for hint in ml_recs.get("algorithm_hints", [])[:3]:
            rec_lines.append(f"  ALGORITHM: {hint}")
        for challenge in ml_recs.get("expected_challenges", [])[:3]:
            rec_lines.append(f"  CHALLENGE: {challenge}")
        for note in ml_recs.get("preprocessing_notes", [])[:3]:
            rec_lines.append(f"  PREPROC: {note}")
        if rec_lines:
            sections.append("ML RECOMMENDATIONS (from data analyst):\n" + "\n".join(rec_lines))

    return "\n\n".join(sections)


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
    tried_algorithms: list = None,
    session_id: str = "",
    analysis_data: dict = None,
) -> dict:
    """Generate baseline ML code, write it, return metadata dict."""
    algorithm_info = select_algorithm(
        task_type, analysis_output, understanding_output, target_column,
        human_feedback, tried_algorithms=tried_algorithms or [],
        analysis_data=analysis_data,
    )
    logger.info(
        "Selected algorithm: %s — %s",
        algorithm_info["algorithm"],
        algorithm_info.get("reason", ""),
    )

    code = generate_ml_code(
        dataset_path, task_type, target_column,
        algorithm_info, understanding_output, analysis_output, human_feedback,
        session_id=session_id,
        analysis_data=analysis_data,
    )
    script_path = GENERATED_CODE_DIR / "step3_ml.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)

    return {
        "script_name": "step3_ml.py",
        "script_path": str(script_path),
        "code": code,
        "algorithm_info": algorithm_info,
        "fix_callback": make_fix_callback(
            script_path,
            dataset_path=dataset_path,
            task_type=task_type,
            target_column=target_column or "",
        ),
    }
