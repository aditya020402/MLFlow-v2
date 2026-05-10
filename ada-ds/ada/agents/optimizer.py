"""Optimization Agent — generates improved ML code for iterations 1-N."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from ada import paths as _paths
from ada.agents.llm_client import client, MODEL
from ada.agents.token_tracker import record as _tok

MAX_TUNE_ITERATIONS    = int(os.environ.get("MAX_TUNE_ITERATIONS", "2"))
MAX_OPTIMIZATION_LOOPS = int(os.environ.get("MAX_OPTIMIZATION_LOOPS", "3"))

_CLASSIFICATION_ALGOS = [
    ("RandomForestClassifier",      {"n_estimators": 100, "random_state": 42},
     {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None]}),
    ("GradientBoostingClassifier",  {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6]}),
    ("LogisticRegression",          {"C": 1.0, "max_iter": 1000, "random_state": 42},
     {"C": [0.01, 0.1, 1.0, 10.0]}),
    ("SVC",                         {"C": 1.0, "kernel": "rbf"},
     {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}),
    ("XGBClassifier",               {"n_estimators": 100, "learning_rate": 0.1, "use_label_encoder": False, "eval_metric": "logloss", "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}),
    ("ExtraTreesClassifier",        {"n_estimators": 100, "random_state": 42},
     {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]}),
    ("AdaBoostClassifier",          {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
     {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}),
    ("KNeighborsClassifier",        {"n_neighbors": 5},
     {"n_neighbors": [3, 5, 7, 11, 15]}),
    ("DecisionTreeClassifier",      {"max_depth": 10, "random_state": 42},
     {"max_depth": [5, 10, 15, 20, None], "min_samples_split": [2, 5, 10]}),
    ("LGBMClassifier",              {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [31, 63, 127]}),
]

_REGRESSION_ALGOS = [
    ("RandomForestRegressor",       {"n_estimators": 100, "random_state": 42},
     {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None]}),
    ("GradientBoostingRegressor",   {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6]}),
    ("Ridge",                       {"alpha": 1.0},
     {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    ("Lasso",                       {"alpha": 1.0},
     {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    ("XGBRegressor",                {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}),
    ("ExtraTreesRegressor",         {"n_estimators": 100, "random_state": 42},
     {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]}),
    ("SVR",                         {"C": 1.0, "kernel": "rbf"},
     {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}),
    ("AdaBoostRegressor",           {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
     {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]}),
    ("KNeighborsRegressor",         {"n_neighbors": 5},
     {"n_neighbors": [3, 5, 7, 11, 15]}),
    ("LGBMRegressor",               {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
     {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [31, 63, 127]}),
]

_CLUSTERING_ALGOS = [
    ("KMeans",                 {"n_clusters": 5, "random_state": 42},
     {"n_clusters": [3, 4, 5, 6, 7]}),
    ("AgglomerativeClustering",{"n_clusters": 4},
     {"n_clusters": [3, 4, 5, 6]}),
    ("GaussianMixture",        {"n_components": 4, "random_state": 42},
     {"n_components": [3, 4, 5, 6]}),
    ("MiniBatchKMeans",        {"n_clusters": 5, "random_state": 42},
     {"n_clusters": [3, 4, 5, 6, 7]}),
]

_META_KEYS = {
    "algorithm", "iteration", "task_type", "model_path",
    "train_samples", "test_samples", "strategy",
}

SYSTEM_PROMPT = """You are an expert ML Optimization Engineer. Write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- For SUPERVISED tasks (classification/regression): use train_test_split(X, y, test_size=0.2, random_state=42).
- For CLUSTERING (unsupervised) tasks:
    * Split X only: X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  — NO y.
    * NEVER invent a target column. NEVER pass y to train_test_split or to any fit() call.
    * NEVER use RandomizedSearchCV for clustering — use a manual itertools.product loop instead.
    * Fit on X_train; evaluate with fit_predict on X_test; metric = silhouette_score (if >1 cluster found).
- Fit ALL preprocessing (scalers, encoders) on TRAIN only — transform both sets.
- Compute ALL metrics on TEST SET ONLY.
- Wrap the final metrics JSON in METRICS_JSON_START / METRICS_JSON_END markers.
- Save the model with pickle to the exact path provided.
- MANDATORY: Save feature column names BEFORE applying any scaler (fit_transform returns a numpy array
  with no .columns). Use this exact pattern:
    feature_names = list(X_train.columns)   # capture BEFORE scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    # ... fit model on X_train_scaled, evaluate on X_test_scaled ...
    features_path = model_path.replace(".pkl", "_features.pkl")
    pickle.dump(feature_names, open(features_path, "wb"))
  NEVER call X_train.columns after fit_transform — it is a numpy array at that point.
- Include self-check: reload model, re-predict on test, warn if metrics diverge.
- SimpleImputer is in sklearn.impute, NOT sklearn.preprocessing.
  Import it as: from sklearn.impute import SimpleImputer
- NEVER use mean_squared_error(..., squared=False) — the `squared` parameter was removed in scikit-learn 1.4.
  Always compute RMSE as: import numpy as np; rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
- VarianceThreshold is in sklearn.feature_selection, NOT sklearn.preprocessing.
  Import it as: from sklearn.feature_selection import VarianceThreshold
- XGBoost / LightGBM LABEL ENCODING: These models require numeric labels for classification.
  ALWAYS encode the target before fitting:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
  Then fit on y_train_enc and predict on y_test_enc.
  Convert predictions back for metrics: y_pred_labels = le.inverse_transform(predictions)
- NEVER pass string labels directly to XGBClassifier, LGBMClassifier, or CatBoostClassifier.
- NEVER use df.pop(col) AND ALSO df.drop(columns=[col]) — they are mutually exclusive.
  To split features and target, use EXACTLY this pattern and nothing else:
    y = df[target_column]
    X = df.drop(columns=[target_column])
  NEVER call df.pop() for this purpose — it mutates df in-place and removes the column before drop() runs.
- NEVER use inplace=True on a column slice — pandas 3.x Copy-on-Write raises ChainedAssignmentError.
  WRONG: df[col].fillna(val, inplace=True)
  RIGHT: df[col] = df[col].fillna(val)
- NEVER use select_dtypes(include=['object']) — deprecated in pandas 3.x.
  Use: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
  Then iterate over cat_cols for encoding.
- NEVER include 'normalize' in LinearRegression hyperparameter grids — it was removed in scikit-learn 1.0.
  Valid LinearRegression params: fit_intercept, copy_X, n_jobs, positive, tol.
- NEVER use pd.np — removed in pandas 2.0. Use numpy (np) directly.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
NEVER use mean_squared_error(..., squared=False) — removed in scikit-learn 1.4.
Compute RMSE as: rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
MANDATORY: Save feature names BEFORE scaling — fit_transform returns a numpy array with no .columns:
    feature_names = list(X_train.columns)   # BEFORE scaler.fit_transform
    X_train_scaled = scaler.fit_transform(X_train)
    # ...
    pickle.dump(feature_names, open(features_path, "wb"))
AttributeError 'numpy.ndarray' has no attribute 'columns': X_train was converted to array by fit_transform.
  Fix: add `feature_names = list(X_train.columns)` before the fit_transform call and use feature_names for pickle.dump.
VarianceThreshold is in sklearn.feature_selection — fix any wrong import of it from sklearn.preprocessing.
SimpleImputer is in sklearn.impute — fix: from sklearn.impute import SimpleImputer (not sklearn.preprocessing).
XGBoost / LightGBM LABEL ENCODING FIX: If error mentions invalid classes or string labels, add:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
  Fit on y_train_enc, predict on y_test_enc, then convert back:
    y_pred_labels = le.inverse_transform(predictions)
  Use y_pred_labels for metrics. NEVER pass string labels directly to XGBClassifier or LGBMClassifier.
KeyError on target column (e.g., "['col'] not found in axis" or "target column not found"):
  The column was removed by df.pop() before drop() ran. Fix: remove ANY df.pop(target_column) call and use:
    y = df[target_column]
    X = df.drop(columns=[target_column])
  NEVER use df.pop() — it mutates df in-place and makes subsequent drop() fail.
NEVER use inplace=True on column slices — use: df[col] = df[col].fillna(val)
NEVER use select_dtypes(include=['object']) — use: cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
NEVER include 'normalize' in LinearRegression param grids — removed in scikit-learn 1.0.
NEVER use pd.np — removed in pandas 2.0. Use numpy directly.
CLUSTERING TASK — KeyError on a column name when code tries `y = df['col']` or uses y_train/y_test:
  This is an unsupervised task. There is NO target column. Fix: remove ALL references to y, y_train, y_test.
  Keep ONLY: X = df[numeric_cols]; X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
  NEVER create a y variable for clustering. NEVER use df.drop() to remove a "target" from features."""


def _format_history_with_metrics(optimization_history: list) -> str:
    entries = []
    for h in optimization_history:
        m = h.get("metrics", {})
        perf = {k: v for k, v in m.items() if k not in _META_KEYS}
        entries.append({
            "iteration":     h.get("iteration"),
            "strategy":      h.get("strategy", "baseline"),
            "algorithm":     h.get("algorithm"),
            "primary_score": h.get("primary_score"),
            "all_metrics":   perf,
            "hyperparameters": m.get("hyperparameters", {}),
        })
    return json.dumps(entries, indent=2, default=str)


def _fallback_new_algo(task_type: str, already_tried: list) -> dict:
    if "classif" in task_type:
        candidates = _CLASSIFICATION_ALGOS
    elif "regress" in task_type:
        candidates = _REGRESSION_ALGOS
    else:
        candidates = _CLUSTERING_ALGOS

    for name, params, space in candidates:
        if name not in already_tried:
            return {
                "strategy": "new_algorithm",
                "algorithm": name,
                "hyperparameters": params,
                "search_space": space,
                "reason": f"Deterministic fallback: {name} selected as next untried algorithm.",
            }
    name, params, space = candidates[-1]
    return {
        "strategy": "new_algorithm",
        "algorithm": name,
        "hyperparameters": params,
        "search_space": space,
        "reason": "All known algorithms have been tried; rerunning last candidate with fresh params.",
    }


def _metrics_fields(task_type: str) -> str:
    if "classif" in task_type:
        return '"accuracy": <float>, "f1": <float>, "precision": <float>, "recall": <float>'
    if "regress" in task_type:
        return '"r2_score": <float>, "rmse": <float>, "mae": <float>  # rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))'
    return '"silhouette_score": <float>'


def _scoring_metric(task_type: str) -> str:
    if "classif" in task_type:
        return "f1_weighted"
    if "regress" in task_type:
        return "r2"
    return "silhouette"


def _scale_needed(algo: str) -> bool:
    return any(kw in algo for kw in ("Logistic", "Ridge", "SV", "Linear", "MLP", "KNN"))


def select_next_strategy(
    task_type: str,
    iteration: int,
    optimization_history: list,
    understanding_output: str,
    best_model: dict,
    current_algo_tune_count: int,
    models_tried: int,
    human_feedback: str = "",
    tried_algorithms: list = None,
    current_algorithm: str = "",
    analysis_data: dict = None,
) -> dict:
    best_algo = best_model.get("algorithm", "unknown")
    best_score = best_model.get("primary_score", 0.0)
    best_metric = best_model.get("primary_metric", "score")

    tune_algo = current_algorithm if current_algorithm else best_algo

    tune_hyperparams = {}
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            tune_hyperparams = (h.get("metrics") or {}).get("hyperparameters", {})
            break
    if not tune_hyperparams and tune_algo == best_algo:
        tune_hyperparams = best_model.get("metrics", {}).get("hyperparameters", {})

    tune_score = best_score
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            tune_score = h.get("primary_score", best_score)
            break

    history_summary = _format_history_with_metrics(optimization_history)

    tune_full_metrics: dict = {}
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            m = h.get("metrics", {})
            tune_full_metrics = {k: v for k, v in m.items() if k not in _META_KEYS}
            break
    if not tune_full_metrics and tune_algo == best_algo:
        m = best_model.get("metrics", {})
        tune_full_metrics = {k: v for k, v in m.items() if k not in _META_KEYS}

    already_tried = tried_algorithms if tried_algorithms is not None else [h.get("algorithm") for h in optimization_history]
    feedback_section = (
        f"\nHuman feedback / instructions:\n{human_feedback}"
        if human_feedback.strip() else ""
    )

    if current_algo_tune_count < MAX_TUNE_ITERATIONS:
        logger.info(
            "Path A — tuning %s (tune round %d/%d, model %d/%d)",
            tune_algo, current_algo_tune_count + 1, MAX_TUNE_ITERATIONS,
            models_tried, MAX_OPTIMIZATION_LOOPS,
        )
        metrics_section = (
            f"  Full metrics from last run: {json.dumps(tune_full_metrics, indent=4, default=str)}"
            if tune_full_metrics else ""
        )
        prompt = f"""You are an ML hyperparameter tuning expert.

Current model to tune: {tune_algo}
  Current {best_metric}: {tune_score:.4f}
  Current hyperparameters: {json.dumps(tune_hyperparams, indent=2)}
{metrics_section}
  Tune round: {current_algo_tune_count + 1} of {MAX_TUNE_ITERATIONS}
  Overall best so far: {best_algo} at {best_score:.4f} {best_metric}

Task type: {task_type}

Full optimization history (all metrics per run):
{history_summary}
{feedback_section}

Provide the hyperparameter search space for RandomizedSearchCV to push {best_metric} beyond {tune_score:.4f}.

Respond ONLY with this JSON:
{{
  "strategy": "tune_best",
  "algorithm": "{tune_algo}",
  "reason": "...",
  "hyperparameters": {json.dumps(tune_hyperparams) if tune_hyperparams else '{"n_estimators": 100}'},
  "search_space": {{"<param>": [<v1>, <v2>, ...], ...}}
}}"""

        system = (
            "You are an ML hyperparameter tuning expert. "
            "Respond ONLY with a JSON object — no markdown, no extra text."
        )
        strategy = "tune_best"
        fallback = {
            "strategy": "tune_best",
            "algorithm": tune_algo,
            "reason": f"Fallback: continuing to tune {tune_algo} (round {current_algo_tune_count + 1}/{MAX_TUNE_ITERATIONS}).",
            "hyperparameters": tune_hyperparams or {"n_estimators": 100, "random_state": 42},
            "search_space": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            },
        }

    else:
        logger.info(
            "Path B — tuning budget exhausted for %s. Selecting new algorithm.",
            tune_algo,
        )
        from ada.agents.ml_engineer import _build_data_context
        data_context = _build_data_context(analysis_data or {}, understanding_output)

        prompt = f"""You are an ML algorithm selection expert.

The current model ({tune_algo}) has been tuned {current_algo_tune_count} time(s) — budget exhausted.
Best score so far: {best_score:.4f} ({best_metric}) achieved by {best_algo}

Task type: {task_type}
Already tried (DO NOT pick these): {already_tried}

=== DATA CHARACTERISTICS ===
{data_context}

=== FULL OPTIMIZATION HISTORY ===
{history_summary}
{feedback_section}

Pick the NEXT ML algorithm most likely to beat {best_score:.4f} ({best_metric}).
{"CLUSTERING TASK: you MUST pick a clustering algorithm. Valid: KMeans, AgglomerativeClustering, GaussianMixture, DBSCAN, MiniBatchKMeans, Birch, SpectralClustering. DO NOT pick supervised algorithms." if not ("classif" in task_type or "regress" in task_type) else "SUPERVISED TASK: do NOT pick clustering algorithms."}

Respond ONLY with this JSON:
{{
  "strategy": "new_algorithm",
  "algorithm": "<algorithm name>",
  "reason": "...",
  "hyperparameters": {{"<param>": <value>, ...}},
  "search_space": {{"<param>": [<v1>, <v2>, ...], ...}}
}}"""

        system = (
            "You are an ML algorithm selection expert. "
            "Respond ONLY with a JSON object — no markdown, no extra text."
        )
        strategy = "new_algorithm"
        if "classif" in task_type:
            fallback = {
                "strategy": "new_algorithm",
                "algorithm": "GradientBoostingClassifier",
                "reason": f"{tune_algo} has been fully tuned. Trying GradientBoosting next.",
                "hyperparameters": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
                "search_space": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6]},
            }
        elif "regress" in task_type:
            fallback = {
                "strategy": "new_algorithm",
                "algorithm": "GradientBoostingRegressor",
                "reason": f"{tune_algo} has been fully tuned. Trying GradientBoosting next.",
                "hyperparameters": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
                "search_space": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6]},
            }
        else:
            fallback = {
                "strategy": "new_algorithm",
                "algorithm": "AgglomerativeClustering",
                "reason": f"{tune_algo} tuning exhausted. Trying AgglomerativeClustering.",
                "hyperparameters": {"n_clusters": 4},
                "search_space": {"n_clusters": [3, 4, 5, 6]},
            }

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    if response.usage:
        _tok("optimizer", response.usage.prompt_tokens, response.usage.completion_tokens)
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    _CLUSTERING_ALGO_NAMES = {
        "KMeans", "MiniBatchKMeans", "AgglomerativeClustering",
        "GaussianMixture", "DBSCAN", "Birch", "SpectralClustering",
        "AffinityPropagation", "MeanShift", "OPTICS",
    }
    _is_cluster = not ("classif" in task_type or "regress" in task_type)

    try:
        result = json.loads(text)
        result["strategy"] = strategy

        if strategy == "new_algorithm":
            chosen = result.get("algorithm", "")

            if _is_cluster and chosen not in _CLUSTERING_ALGO_NAMES:
                logger.warning("Optimizer picked supervised algorithm '%s' for clustering — substituting.", chosen)
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)

            elif not _is_cluster and chosen in _CLUSTERING_ALGO_NAMES:
                logger.warning("Optimizer picked clustering algorithm '%s' for supervised task — substituting.", chosen)
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)

            elif chosen in already_tried:
                logger.warning("LLM picked already-tried algorithm '%s'; substituting.", chosen)
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)

        return result
    except json.JSONDecodeError:
        logger.warning("Could not parse optimizer JSON; using fallback.")
        if strategy == "new_algorithm" and fallback.get("algorithm") in already_tried:
            return _fallback_new_algo(task_type, already_tried)
        return fallback


def _build_preprocessing_intelligence(analysis_data: dict, algo: str, task_type: str) -> str:
    if not analysis_data:
        return "No structured analysis data available — apply standard preprocessing heuristics."

    sections = []

    col_types = analysis_data.get("col_types", {})
    if isinstance(col_types, dict) and col_types:
        by_type: dict[str, list] = {}
        for col, ctype in col_types.items():
            by_type.setdefault(ctype, []).append(col)
        type_lines = "\n".join(f"  {ctype}: {', '.join(str(c) for c in cols)}" for ctype, cols in by_type.items())
        sections.append(f"COLUMN TYPES:\n{type_lines}")

    enc = analysis_data.get("encoding_recommendations", {})
    if isinstance(enc, dict) and enc:
        enc_lines = "\n".join(f"  {col}: {how}" for col, how in enc.items())
        sections.append(f"ENCODING RECOMMENDATIONS (apply exactly):\n{enc_lines}")
    elif isinstance(enc, str) and enc:
        sections.append(f"ENCODING RECOMMENDATIONS: {enc}")
    elif isinstance(enc, list) and enc:
        sections.append("ENCODING RECOMMENDATIONS:\n" + "\n".join(f"  {e}" for e in enc))

    sections.append(
        f"SCALING DECISION: The algorithm is {algo}. "
        f"Tree-based models do NOT need scaling. Distance-based and linear models DO need scaling."
    )

    hcp = analysis_data.get("high_correlation_pairs", [])
    if isinstance(hcp, list) and hcp:
        hcp_strs = [item["pair"] if isinstance(item, dict) else str(item) for item in hcp]
        sections.append(f"HIGH-CORRELATION PAIRS (drop one from each after encoding):\n  {', '.join(hcp_strs)}")

    imb_rec = analysis_data.get("imbalance_recommendation", "")
    if imb_rec and imb_rec != "n/a":
        ratio = analysis_data.get("imbalance_ratio", 1.0)
        dist = analysis_data.get("class_distribution", {})
        if isinstance(dist, dict):
            dist_str = ", ".join(f"{k}: {v}" for k, v in list(dist.items())[:6])
        else:
            dist_str = str(dist)
        sections.append(
            f"CLASS IMBALANCE:\n"
            f"  Imbalance ratio: {ratio:.1f}x\n"
            f"  Class distribution: {{{dist_str}}}\n"
            f"  Recommendation: {imb_rec}"
        )
    else:
        sections.append("CLASS IMBALANCE: balanced — no special handling needed")

    top = analysis_data.get("top_features", [])
    if isinstance(top, list) and top:
        sections.append(f"TOP PREDICTIVE FEATURES:\n  {', '.join(str(x) for x in top[:10])}")
    elif isinstance(top, dict) and top:
        sections.append(f"TOP PREDICTIVE FEATURES:\n  {', '.join(str(k) for k in list(top.keys())[:10])}")

    return "\n\n".join(sections)


def generate_optimization_code(
    dataset_path: str,
    task_type: str,
    target_column: str,
    iteration: int,
    strategy: dict,
    optimization_history: list,
    best_model: dict,
    human_feedback: str = "",
    last_working_code: str = "",
    session_id: str = "",
    analysis_data: dict = None,
) -> str:
    dataset_path = dataset_path.replace("\\", "/")
    algo = strategy["algorithm"]
    strat_type = strategy.get("strategy", "new_algorithm")
    hyperparams = json.dumps(strategy.get("hyperparameters", {}))
    search_space = json.dumps(strategy.get("search_space", {}))
    reason = strategy.get("reason", "")
    session_out = _paths.OUTPUTS_DIR / (session_id if session_id else "default")
    session_out.mkdir(parents=True, exist_ok=True)
    model_path = str(session_out / f"model_iter{iteration}.pkl").replace("\\", "/")
    is_cluster = not ("classif" in task_type or "regress" in task_type)
    stratify = "stratify=y, " if "classif" in task_type else ""
    scale = _scale_needed(algo)
    scoring = _scoring_metric(task_type)
    x_fit  = f"X_train{'_scaled' if scale else ''}"
    x_eval = f"X_test{'_scaled' if scale else ''}"

    baseline_score = best_model.get("primary_score", 0.0)
    baseline_algo = best_model.get("algorithm", "unknown")
    primary_metric = best_model.get("primary_metric", "score")

    history_summary = json.dumps(
        [
            {
                "iteration": h.get("iteration"),
                "strategy": h.get("strategy", "baseline"),
                "algorithm": h.get("algorithm"),
                "score": h.get("primary_score"),
            }
            for h in optimization_history
        ],
        indent=2,
    )

    feedback_section = (
        f"\nHuman feedback / instructions:\n{human_feedback}"
        if human_feedback.strip() else ""
    )

    preproc_intel = _build_preprocessing_intelligence(analysis_data or {}, algo, task_type)

    if strat_type == "tune_best" and last_working_code.strip() and not is_cluster:
        logger.info("Optimizer [iter %d] tune_best: adapting previous working code as template.", iteration)
        template_prompt = f"""You have a working Python ML script that successfully trained {algo}.
Adapt it for tuning round {iteration} by updating ONLY the hyperparameter search space and metadata.

PREVIOUS WORKING CODE:
{last_working_code}

Changes to make — update THESE values and NOTHING ELSE:
1. param_distributions / search_space -> {search_space}
2. Base hyperparameters passed to {algo}(...) -> {hyperparams}
3. Model save path -> "{model_path}"
4. "iteration" value in the metrics dict -> {iteration}
5. RandomizedSearchCV n_iter -> 20, cv -> 5  (keep scoring='{scoring}', n_jobs=-1, random_state=42)
6. Baseline to beat comment -> {baseline_score:.4f} ({primary_metric}) from {baseline_algo}

Keep EVERYTHING ELSE byte-for-byte identical.

Output ONLY the complete adapted Python code."""

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": template_prompt},
            ],
        )
        if response.usage:
            _tok("optimizer", response.usage.prompt_tokens, response.usage.completion_tokens)
        code = _strip_markdown(response.choices[0].message.content.strip())
        logger.info("Optimizer [iter %d] tune_best template adapted for %s", iteration, algo)
        return code

    if is_cluster:
        n_iter = 20 if strat_type == "tune_best" else 10
        tuning_section = f"""5. CLUSTERING HYPERPARAMETER SEARCH (NO RandomizedSearchCV — use manual itertools loop):
   - Base hyperparameters: {hyperparams}
   - Search space: {search_space}
   - Use itertools.product to iterate over search space, fit each candidate on {x_fit},
     evaluate silhouette_score on {x_eval} via fit_predict, keep the best.
   - Baseline to beat: {baseline_score:.4f} ({primary_metric}) from {baseline_algo}
   - Save best_model_instance to: {model_path}"""
    elif strat_type == "tune_best":
        tuning_section = f"""5. HYPERPARAMETER TUNING — PATH A (tune_best):
   - Base hyperparameters: {hyperparams}
   - Search space: {search_space}
   - Use RandomizedSearchCV(n_iter=20, cv=5, scoring='{scoring}', n_jobs=-1, random_state=42, refit=True)
   - Evaluate best_model_instance on {x_eval} (TEST ONLY)
   - Baseline to beat: {baseline_score:.4f} ({primary_metric}) from {baseline_algo}
   - Save best_model_instance to: {model_path}"""
    else:
        tuning_section = f"""5. NEW ALGORITHM + QUICK HYPERPARAMETER SEARCH — PATH B (new_algorithm):
   - Initial hyperparameters: {hyperparams}
   - Search space: {search_space}
   - Use RandomizedSearchCV(n_iter=10, cv=3, scoring='{scoring}', n_jobs=-1, random_state=42, refit=True)
   - Baseline to beat: {baseline_score:.4f} ({primary_metric}) achieved by {baseline_algo}
   - Evaluate best_model_instance on {x_eval} (TEST ONLY)
   - Save best_model_instance to: {model_path}"""

    prompt = f"""Write Python code for optimization iteration {iteration} — Strategy: {strat_type}.

Dataset: {dataset_path}
Task: {task_type}
{"Target: " + target_column if target_column else "Unsupervised"}
Algorithm: {algo}
Strategy: {strat_type}
Reason: {reason}
{feedback_section}

Optimization history:
{history_summary}

=== PREPROCESSING INTELLIGENCE ===
{preproc_intel}

Requirements (follow EXACTLY):
1. pandas read_csv to load dataset
2. SMART PREPROCESSING (use intelligence section above)
   a. Drop columns where missing > 50%
   b. Handle missing values: median/mean for numeric, mode for categorical
   c. Apply encoding_recommendations exactly
   d. Drop near-zero variance features; drop one from each high-correlation pair
   e. Apply class imbalance strategy if recommended
   {"f. Separate X and y (target='" + target_column + "')" if target_column else "f. X = all columns — unsupervised"}
{"3. CLUSTERING SPLIT — NO y:" if is_cluster else "3. SUPERVISED SPLIT:"}
   {"X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)" if is_cluster else f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, {stratify})"}
4. {"StandardScaler: fit X_train, transform both -> X_train_scaled, X_test_scaled" if scale else "No scaling needed (tree-based model)"}
{tuning_section}
6. Compute on TEST SET ONLY:
   {"predictions = best_model_instance.fit_predict(" + x_eval + ")  # clustering" if is_cluster else "predictions = best_model_instance.predict(" + x_eval + ")"}
   {_metrics_fields(task_type)}
7. pickle.dump(best_model_instance, open("{model_path}", "wb"))
8. SELF-CHECK: sc_model = pickle.load(open("{model_path}", "rb")); {"sc_preds = sc_model.fit_predict(" + x_eval + ")" if is_cluster else "sc_preds = sc_model.predict(" + x_eval + ")"}; warn if sc_preds != predictions
9. Print metrics EXACTLY as:

print("METRICS_JSON_START")
print(json.dumps(metrics, indent=2, default=str))
print("METRICS_JSON_END")

metrics dict:
{{
  "algorithm": "{algo}",
  "strategy": "{strat_type}",
  "iteration": {iteration},
  "task_type": "{task_type}",
  "model_path": "{model_path}",
  "train_samples": <int>,
  "test_samples": <int>,
  "hyperparameters": best_params,
  {_metrics_fields(task_type)}
}}

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    if response.usage:
        _tok("optimizer", response.usage.prompt_tokens, response.usage.completion_tokens)
    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Optimizer [iter %d] strategy=%s algorithm=%s", iteration, strat_type, algo)
    return code


def fix_optimization_code(
    current_code: str,
    stderr: str,
    stdout: str,
    attempt: int,
    dataset_path: str = "",
    task_type: str = "",
    target_column: str = "",
) -> str:
    logger.info("Optimizer fixing code (attempt %d)...", attempt)

    context_lines = []
    if dataset_path:
        context_lines.append(f"Dataset path: {dataset_path}")
    if task_type:
        context_lines.append(f"Task type: {task_type}")
    if target_column:
        context_lines.append(
            f"Target column: '{target_column}' — use this exact column name, do NOT hardcode 'target' or guess."
        )
    context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""

    prompt = f"""Code failed.

{context_block}ERROR:
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
            {"role": "user", "content": prompt},
        ],
    )
    if response.usage:
        _tok("optimizer", response.usage.prompt_tokens, response.usage.completion_tokens)
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(
    current_code_path: Path,
    dataset_path: str = "",
    task_type: str = "",
    target_column: str = "",
):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_optimization_code(
            current_code_path.read_text(encoding="utf-8"), stderr, stdout, attempt,
            dataset_path=dataset_path,
            task_type=task_type,
            target_column=target_column,
        )
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
    iteration: int,
    optimization_history: list,
    understanding_output: str,
    best_model: dict,
    current_algo_tune_count: int = 0,
    models_tried: int = 0,
    human_feedback: str = "",
    tried_algorithms: list = None,
    last_working_code: str = "",
    current_algorithm: str = "",
    session_id: str = "",
    analysis_data: dict = None,
) -> dict:
    strategy = select_next_strategy(
        task_type, iteration, optimization_history,
        understanding_output, best_model,
        current_algo_tune_count, models_tried,
        human_feedback,
        tried_algorithms=tried_algorithms or [],
        current_algorithm=current_algorithm,
        analysis_data=analysis_data,
    )
    logger.info(
        "Optimizer [iter %d] strategy=%s algorithm=%s — %s",
        iteration,
        strategy.get("strategy"),
        strategy["algorithm"],
        strategy.get("reason", ""),
    )

    code = generate_optimization_code(
        dataset_path, task_type, target_column,
        iteration, strategy, optimization_history, best_model, human_feedback,
        last_working_code=last_working_code if strategy.get("strategy") == "tune_best" else "",
        session_id=session_id,
        analysis_data=analysis_data,
    )

    script_name = f"step3_ml_iter{iteration}.py"
    script_path = _paths.GENERATED_CODE_DIR / script_name
    script_path.write_text(code, encoding="utf-8")
    logger.info("Written: %s", script_path)

    return {
        "script_name": script_name,
        "script_path": str(script_path),
        "code": code,
        "algorithm_info": strategy,
        "strategy": strategy.get("strategy", "new_algorithm"),
        "fix_callback": make_fix_callback(
            script_path,
            dataset_path=dataset_path,
            task_type=task_type,
            target_column=target_column or "",
        ),
    }
