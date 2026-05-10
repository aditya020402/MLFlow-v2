"""Optimization Agent — generates improved ML code for iterations 1-N.

Two paths per call:
  PATH A (tune_best)    — RandomizedSearchCV around the current best model's hyperparameters.
  PATH B (new_algorithm) — Train a new algorithm with a quick hyperparameter search so it
                           has its own optimised set of params before being compared.

Each path saves exactly ONE pkl per iteration (the best model found by the search).
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

from agents.llm_client import client, MODEL
from agents.token_tracker import record as _tok

MAX_TUNE_ITERATIONS    = int(os.environ.get("MAX_TUNE_ITERATIONS", "2"))
MAX_OPTIMIZATION_LOOPS = int(os.environ.get("MAX_OPTIMIZATION_LOOPS", "3"))

# Ordered lists used to deterministically substitute when LLM picks a duplicate algo.
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


def _format_history_with_metrics(optimization_history: list) -> str:
    """Full history including per-run metric breakdown, hyperparameters used, and strategy."""
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
    """Return the first untried algorithm from the ordered defaults list."""
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
    # All known algos exhausted — return last one anyway
    name, params, space = candidates[-1]
    return {
        "strategy": "new_algorithm",
        "algorithm": name,
        "hyperparameters": params,
        "search_space": space,
        "reason": "All known algorithms have been tried; rerunning last candidate with fresh params.",
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
- Include self-check: reload model, re-predict on test, warn if metrics diverge.
- NEVER use mean_squared_error(..., squared=False) — the `squared` parameter was removed in scikit-learn 1.4.
  Always compute RMSE as: import numpy as np; rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks.
NEVER use mean_squared_error(..., squared=False) — removed in scikit-learn 1.4.
Compute RMSE as: rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))"""

STRATEGY_SELECTION_PROMPT = """You are an ML optimization strategist.
Respond ONLY with a JSON object — no markdown, no extra text.

For PATH A (tune_best):
{
  "strategy": "tune_best",
  "algorithm": "<same algorithm as best model>",
  "reason": "2-3 sentence rationale: (1) what the current results show and why further tuning is worthwhile, (2) which hyperparameter ranges you will explore and why, (3) what improvement you expect.",
  "hyperparameters": {"<base param>": <value>, ...},
  "search_space": {"<param>": [<v1>, <v2>, ...], ...}
}

For PATH B (new_algorithm):
{
  "strategy": "new_algorithm",
  "algorithm": "<new algorithm name>",
  "reason": "2-3 sentence rationale: (1) why the previous model(s) have been exhausted or are insufficient, (2) which specific characteristics of the data or past results make this new algorithm a better fit, (3) what improvement over the current best score you expect.",
  "hyperparameters": {"<initial param>": <value>, ...},
  "search_space": {"<param>": [<v1>, <v2>, ...], ...}
}"""


def _metrics_fields(task_type: str) -> str:
    if "classif" in task_type:
        return '"accuracy": <float>, "f1": <float>, "precision": <float>, "recall": <float>'
    if "regress" in task_type:
        return '"r2_score": <float>, "rmse": <float>, "mae": <float>  # rmse = float(np.sqrt(mean_squared_error(y_test, predictions))) — NEVER use squared=False'
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
    """Deterministically pick PATH A or PATH B from the graph-level counters.

    PATH A (tune_best)    — current model still has tuning budget remaining.
    PATH B (new_algorithm) — tuning budget exhausted; move to a new ML algorithm.

    The strategy choice is NOT delegated to the LLM. The LLM is only asked to fill
    in the *details*: which hyperparameter search space (Path A) or which new
    algorithm (Path B), plus a rationale.
    """
    best_algo = best_model.get("algorithm", "unknown")
    best_score = best_model.get("primary_score", 0.0)
    best_metric = best_model.get("primary_metric", "score")

    # tune_best tunes the algorithm we're currently focusing on, not necessarily
    # the all-time best. This ensures a new_algorithm gets its own tuning rounds
    # before we compare it against the historical best.
    tune_algo = current_algorithm if current_algorithm else best_algo

    # Find that algorithm's most recent hyperparameters from history (or fall
    # back to best_model's if it IS the best model, or empty dict).
    tune_hyperparams = {}
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            tune_hyperparams = (h.get("metrics") or {}).get("hyperparameters", {})
            break
    if not tune_hyperparams and tune_algo == best_algo:
        tune_hyperparams = best_model.get("metrics", {}).get("hyperparameters", {})

    # Score and metric for the algo we're tuning (may differ from all-time best)
    tune_score = best_score
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            tune_score = h.get("primary_score", best_score)
            break

    history_summary = _format_history_with_metrics(optimization_history)

    # Find the full metrics profile from the last run of tune_algo (for tune_best reasoning)
    tune_full_metrics: dict = {}
    for h in reversed(optimization_history):
        if h.get("algorithm") == tune_algo:
            m = h.get("metrics", {})
            tune_full_metrics = {k: v for k, v in m.items() if k not in _META_KEYS}
            break
    if not tune_full_metrics and tune_algo == best_algo:
        m = best_model.get("metrics", {})
        tune_full_metrics = {k: v for k, v in m.items() if k not in _META_KEYS}

    # Use the authoritative state-tracked list; fall back to deriving from history.
    already_tried = tried_algorithms if tried_algorithms is not None else [h.get("algorithm") for h in optimization_history]
    feedback_section = (
        f"\nHuman feedback / instructions:\n{human_feedback}"
        if human_feedback.strip() else ""
    )

    # ── PATH A: tune the current algorithm ────────────────────────
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

Look at ALL the metric values above — not just the primary score — to decide which hyperparameter
dimensions have room to improve. For example:
- If precision is high but recall is low → adjust class_weight, threshold, or depth
- If training was fast but score is mediocre → increase n_estimators / depth
- If scores plateau between rounds → try a wider / different search range

Provide the hyperparameter search space for RandomizedSearchCV to push {best_metric} beyond {tune_score:.4f}.

For the `reason` field write 2-3 sentences explaining:
1. What the full metric profile shows (cite specific numbers) and which dimensions have the most room to improve.
2. Why you chose these specific search ranges based on those numbers.
3. What score improvement you expect and why.

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
            "reason": (
                f"Fallback: continuing to tune {tune_algo} (round {current_algo_tune_count + 1}/"
                f"{MAX_TUNE_ITERATIONS}). Exploring wider hyperparameter ranges to push beyond "
                f"{tune_score:.4f} {best_metric}."
            ),
            "hyperparameters": tune_hyperparams or {"n_estimators": 100, "random_state": 42},
            "search_space": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            },
        }

    # ── PATH B: move to a new ML algorithm ────────────────────────
    else:
        logger.info(
            "Path B — tuning budget exhausted for %s (%d/%d). Selecting new algorithm (model %d→%d/%d).",
            tune_algo, current_algo_tune_count, MAX_TUNE_ITERATIONS,
            models_tried, models_tried + 1, MAX_OPTIMIZATION_LOOPS,
        )
        from agents.ml_engineer import _build_data_context
        data_context = _build_data_context(analysis_data or {}, understanding_output)

        prompt = f"""You are an ML algorithm selection expert.

The current model ({tune_algo}) has been tuned {current_algo_tune_count} time(s) — budget exhausted.
Best score so far: {best_score:.4f} ({best_metric}) achieved by {best_algo}

Task type: {task_type}
Already tried (DO NOT pick these): {already_tried}

=== DATA CHARACTERISTICS (reason from these — do not ignore) ===
{data_context}

=== FULL OPTIMIZATION HISTORY (all metrics per run — use these to understand what worked and what didn't) ===
{history_summary}
{feedback_section}

Study the full metric breakdowns above: which algorithms scored well on primary metric but failed on
secondary ones? Where did performance plateau? Use these observations to pick an algorithm that
addresses the weaknesses the history reveals.

Pick the NEXT ML algorithm most likely to beat {best_score:.4f} ({best_metric}).
{"CLUSTERING TASK: you MUST pick a clustering algorithm. Valid: KMeans, AgglomerativeClustering, GaussianMixture, DBSCAN, MiniBatchKMeans, Birch, SpectralClustering. DO NOT pick supervised algorithms." if not ("classif" in task_type or "regress" in task_type) else "SUPERVISED TASK: do NOT pick clustering algorithms."}

Reason from the actual data characteristics AND the full metric history above:
- Dataset size affects which algorithms are feasible
- Feature types (numeric/categorical) constrain algorithm choice
- Outlier presence favours robust algorithms
- Class imbalance affects which algorithms need class_weight
- Distance-based algorithms (KNN, SVM, DBSCAN, KMeans) REQUIRE StandardScaler on this data
- Previous results show which model families are working and which aren't

Provide both `hyperparameters` (initial values) and `search_space` (for a quick RandomizedSearchCV).

For the `reason` field write 2-3 sentences explaining:
1. Why the tried algorithms have reached their limit — cite specific metric values from history.
2. Which specific data characteristics AND past result patterns make this new algorithm a better fit.
3. What score improvement over {best_score:.4f} ({best_metric}) you expect and why.

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
                "reason": (
                    f"{tune_algo} has been fully tuned ({current_algo_tune_count} rounds). "
                    "GradientBoosting often captures non-linear patterns missed by other ensembles. "
                    f"Expect to push past {best_score:.4f} {best_metric} with careful learning-rate tuning."
                ),
                "hyperparameters": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
                "search_space": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5, 6],
                },
            }
        elif "regress" in task_type:
            fallback = {
                "strategy": "new_algorithm",
                "algorithm": "GradientBoostingRegressor",
                "reason": (
                    f"{tune_algo} has been fully tuned ({current_algo_tune_count} rounds). "
                    "GradientBoosting handles complex feature interactions and often improves R². "
                    f"Targeting {best_metric} improvement beyond {best_score:.4f}."
                ),
                "hyperparameters": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
                "search_space": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5, 6],
                },
            }
        else:
            fallback = {
                "strategy": "new_algorithm",
                "algorithm": "AgglomerativeClustering",
                "reason": (
                    f"{tune_algo} tuning exhausted. "
                    "AgglomerativeClustering uses a different linkage-based approach "
                    "that may reveal cluster structures KMeans missed."
                ),
                "hyperparameters": {"n_clusters": 4},
                "search_space": {"n_clusters": [3, 4, 5, 6]},
            }

    # ── LLM call (fills in details; strategy already decided above) ──
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

    _CLUSTERING_ALGOS = {
        "KMeans", "MiniBatchKMeans", "AgglomerativeClustering",
        "GaussianMixture", "DBSCAN", "Birch", "SpectralClustering",
        "AffinityPropagation", "MeanShift", "OPTICS",
    }
    _is_cluster = not ("classif" in task_type or "regress" in task_type)

    try:
        result = json.loads(text)
        result["strategy"] = strategy   # always override — LLM does not decide this

        if strategy == "new_algorithm":
            chosen = result.get("algorithm", "")

            # Hard guard: clustering task must only use clustering algorithms
            if _is_cluster and chosen not in _CLUSTERING_ALGOS:
                logger.warning(
                    "Optimizer picked supervised algorithm '%s' for clustering task — substituting.", chosen
                )
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)
                result["reason"] = (
                    f"LLM suggested '{chosen}' which is a supervised algorithm — invalid for clustering. "
                    + sub["reason"]
                )

            # Hard guard: supervised task must not use clustering algorithms
            elif not _is_cluster and chosen in _CLUSTERING_ALGOS:
                logger.warning(
                    "Optimizer picked clustering algorithm '%s' for supervised task — substituting.", chosen
                )
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)
                result["reason"] = (
                    f"LLM suggested '{chosen}' which is a clustering algorithm — invalid for supervised task. "
                    + sub["reason"]
                )

            # Standard duplicate guard
            elif chosen in already_tried:
                logger.warning(
                    "LLM picked already-tried algorithm '%s'; substituting from defaults.", chosen
                )
                sub = _fallback_new_algo(task_type, already_tried)
                result.update(sub)
                result["reason"] = (
                    f"LLM suggested '{chosen}' which was already tried. "
                    + sub["reason"]
                )

        return result
    except json.JSONDecodeError:
        logger.warning("Could not parse optimizer JSON; using fallback.")
        if strategy == "new_algorithm" and fallback.get("algorithm") in already_tried:
            return _fallback_new_algo(task_type, already_tried)
        return fallback


def _build_preprocessing_intelligence(analysis_data: dict, algo: str, task_type: str) -> str:
    """Format structured analysis_data into an LLM-readable preprocessing context."""
    if not analysis_data:
        return "No structured analysis data available — apply standard preprocessing heuristics."

    sections = []

    col_types = analysis_data.get("col_types", {})
    if col_types:
        by_type: dict[str, list] = {}
        for col, ctype in col_types.items():
            by_type.setdefault(ctype, []).append(col)
        type_lines = "\n".join(f"  {ctype}: {', '.join(cols)}" for ctype, cols in by_type.items())
        sections.append(f"COLUMN TYPES:\n{type_lines}")

    enc = analysis_data.get("encoding_recommendations", {})
    if enc:
        enc_lines = "\n".join(f"  {col}: {how}" for col, how in enc.items())
        sections.append(f"ENCODING RECOMMENDATIONS (apply exactly):\n{enc_lines}")

    sections.append(
        f"SCALING DECISION: The algorithm is {algo}. "
        f"Decide whether StandardScaler is needed based on {algo}'s mathematical properties. "
        f"Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees, Decision Tree, AdaBoost) "
        f"do NOT need scaling. Distance-based models (KNN, SVM, SVR, DBSCAN, KMeans, AgglomerativeClustering), "
        f"linear models (Logistic Regression, Ridge, Lasso, ElasticNet), and neural networks DO need scaling. "
        f"GaussianMixture needs scaling. Apply StandardScaler if needed."
    )

    hcp = analysis_data.get("high_correlation_pairs", [])
    if hcp:
        hcp_strs = [item["pair"] if isinstance(item, dict) else str(item) for item in hcp]
        sections.append(f"HIGH-CORRELATION PAIRS (drop one from each after encoding):\n  {', '.join(hcp_strs)}")

    outlier = analysis_data.get("outlier_summary", {})
    capped = [(col, v.get("action", "")) for col, v in outlier.items() if "capped" in v.get("action", "")]
    if capped:
        cap_lines = "\n".join(f"  {col}: {action}" for col, action in capped)
        sections.append(f"OUTLIER HANDLING (already applied to cleaned CSV — do NOT re-apply):\n{cap_lines}")

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

    top = analysis_data.get("top_features", [])
    if top:
        sections.append(f"TOP PREDICTIVE FEATURES:\n  {', '.join(top[:10])}")

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
    """Return code string for the given iteration."""
    algo = strategy["algorithm"]
    strat_type = strategy.get("strategy", "new_algorithm")
    hyperparams = json.dumps(strategy.get("hyperparameters", {}))
    search_space = json.dumps(strategy.get("search_space", {}))
    reason = strategy.get("reason", "")
    session_out = OUTPUTS_DIR / (session_id if session_id else "default")
    session_out.mkdir(parents=True, exist_ok=True)
    model_path = str(session_out / f"model_iter{iteration}.pkl")
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

    # ── PATH A with working code — template adaptation (avoids repeating past bugs) ──
    # Skip for clustering: previous code used manual grid search (no RandomizedSearchCV),
    # so it's safer to always generate fresh clustering code from the explicit prompt below.
    if strat_type == "tune_best" and last_working_code.strip() and not is_cluster:
        logger.info(
            "Optimizer [iter %d] tune_best: adapting previous working code as template.", iteration
        )
        template_prompt = f"""You have a working Python ML script that successfully trained {algo}.
Adapt it for tuning round {iteration} by updating ONLY the hyperparameter search space and metadata.

PREVIOUS WORKING CODE:
{last_working_code}

Changes to make — update THESE values and NOTHING ELSE:
1. param_distributions / search_space → {search_space}
2. Base hyperparameters passed to {algo}(...) → {hyperparams}
3. Model save path → "{model_path}"
4. "iteration" value in the metrics dict → {iteration}
5. RandomizedSearchCV n_iter → 20, cv → 5  (keep scoring='{scoring}', n_jobs=-1, random_state=42)
6. Baseline to beat comment → {baseline_score:.4f} ({primary_metric}) from {baseline_algo}

Keep EVERYTHING ELSE byte-for-byte identical:
- All imports
- Data loading (pd.read_csv)
- All preprocessing steps (missing-value handling, encoding, scaling)
- train_test_split parameters
- Metric computation code
- The METRICS_JSON_START / METRICS_JSON_END print block
- The self-check (pickle reload + re-predict)
- All variable names and structure

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

    # ── PATH A without working code OR PATH B — generate from scratch ──────
    if is_cluster:
        # Clustering: manual grid search — RandomizedSearchCV cannot score clustering correctly
        n_iter = 20 if strat_type == "tune_best" else 10
        tuning_section = f"""5. CLUSTERING HYPERPARAMETER SEARCH (NO RandomizedSearchCV — use manual itertools loop):
   RandomizedSearchCV requires y and a scorer incompatible with clustering — do NOT use it.
   Instead, iterate over the search space manually, fit each candidate on {x_fit},
   evaluate silhouette_score on {x_eval} via fit_predict, keep the best.
   - {"Base" if strat_type == "tune_best" else "Initial"} hyperparameters: {hyperparams}
   - Search space: {search_space}
   - Code pattern to follow EXACTLY:
       import itertools
       from sklearn.metrics import silhouette_score as _sil_score
       param_grid = {search_space}
       keys = list(param_grid.keys())
       vals  = list(param_grid.values())
       best_sil = -2.0
       best_params = {hyperparams}
       best_model_instance = {algo}(**{hyperparams})
       best_model_instance.fit({x_fit})
       for combo in itertools.product(*vals):
           p = dict(zip(keys, combo))
           try:
               candidate = {algo}(**{{**{hyperparams}, **p}})
               candidate.fit({x_fit})
               lbl = candidate.fit_predict({x_eval})
               if len(set(lbl)) > 1:
                   s = _sil_score({x_eval}, lbl)
                   if s > best_sil:
                       best_sil, best_params, best_model_instance = s, p, candidate
           except Exception:
               pass
       predictions = best_model_instance.fit_predict({x_eval})
       silhouette_score_val = _sil_score({x_eval}, predictions) if len(set(predictions)) > 1 else -1.0
   - Baseline to beat: {baseline_score:.4f} ({primary_metric}) from {baseline_algo}
   - Save best_model_instance to: {model_path} — ONE pkl only"""
    elif strat_type == "tune_best":
        tuning_section = f"""5. HYPERPARAMETER TUNING — PATH A (tune_best):
   This iteration tunes the current best model via RandomizedSearchCV.
   - Base hyperparameters (starting point): {hyperparams}
   - Search space: {search_space}
   - Code pattern to follow EXACTLY:
       from sklearn.model_selection import RandomizedSearchCV
       import scipy.stats as stats  # use if you need distributions
       param_distributions = {search_space}
       base_est = {algo}(**{hyperparams})
       search = RandomizedSearchCV(
           base_est, param_distributions,
           n_iter=20, cv=5, scoring='{scoring}',
           n_jobs=-1, random_state=42, refit=True
       )
       search.fit({x_fit}, y_train)
       best_model_instance = search.best_estimator_
       best_params = search.best_params_
   - Evaluate best_model_instance on {x_eval} (TEST ONLY)
   - Baseline to beat: {baseline_score:.4f} ({primary_metric}) from {baseline_algo}
   - Save best_model_instance (the tuned estimator) to: {model_path} — ONE pkl only"""
    else:
        tuning_section = f"""5. NEW ALGORITHM + QUICK HYPERPARAMETER SEARCH — PATH B (new_algorithm):
   This iteration trains a new algorithm with its own hyperparameter search.
   - Initial hyperparameters: {hyperparams}
   - Search space: {search_space}
   - Code pattern to follow EXACTLY:
       from sklearn.model_selection import RandomizedSearchCV
       param_distributions = {search_space}
       base_est = {algo}(**{hyperparams})
       search = RandomizedSearchCV(
           base_est, param_distributions,
           n_iter=10, cv=3, scoring='{scoring}',
           n_jobs=-1, random_state=42, refit=True
       )
       search.fit({x_fit}, y_train)
       best_model_instance = search.best_estimator_
       best_params = search.best_params_
   - BASELINE TO BEAT: {baseline_score:.4f} ({primary_metric}) achieved by {baseline_algo}
   - Evaluate best_model_instance on {x_eval} (TEST ONLY)
   - Save best_model_instance to: {model_path} — ONE pkl for this algorithm, the best found"""

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

=== PREPROCESSING INTELLIGENCE (from data analysis — follow these decisions exactly) ===
{preproc_intel}

Requirements (follow EXACTLY):
1. pandas read_csv to load dataset

2. SMART PREPROCESSING (use intelligence section above):
   a. Drop columns where missing > 50%
   b. Handle remaining missing values: median/mean for numeric (based on skewness), mode for categorical
   c. ENCODING — apply exactly per encoding_recommendations above:
      - "label": LabelEncoder (fit on train only, transform both)
      - "onehot": pd.get_dummies on train, reindex test to match train columns
      - "frequency": map each category to its frequency in TRAIN set only
      - "target": map each category to mean target value in TRAIN set only
      - "drop": drop the column entirely
   d. FEATURE SELECTION — drop near-zero variance features (std < 0.001 after encoding);
      drop one column from each high-correlation pair listed above
   e. CLASS IMBALANCE — if imbalance_recommendation says class_weight or SMOTE:
      apply class_weight='balanced' if {algo} supports it, else SMOTE (fit on train only)
   {"f. Separate X and y (target='" + target_column + "')" if target_column else "f. X = all columns — unsupervised task, NO target column, do NOT invent one"}

{"3. CLUSTERING SPLIT — NO y:" if is_cluster else "3. SUPERVISED SPLIT:"}
   {"X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # NO y argument — this is unsupervised" if is_cluster else f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, {stratify})"}
4. {"StandardScaler: fit X_train, transform X_train + X_test → X_train_scaled, X_test_scaled" if scale else "No scaling needed (tree-based model per intelligence section)"}
{tuning_section}
6. Compute on TEST SET ONLY:
   {"predictions = best_model_instance.fit_predict(" + x_eval + ")  # clustering: fit_predict, NOT predict" if is_cluster else "predictions = best_model_instance.predict(" + x_eval + ")"}
   {_metrics_fields(task_type)}
   {"silhouette_score_val already computed in step 5 — use it directly in metrics dict" if is_cluster else ""}
7. pickle.dump(best_model_instance, open("{model_path}", "wb"))
8. SELF-CHECK (data leakage guard):
   sc_model = pickle.load(open("{model_path}", "rb"))
   {"sc_preds = sc_model.fit_predict(" + x_eval + ")  # clustering: use fit_predict" if is_cluster else "sc_preds = sc_model.predict(" + x_eval + ")"}
   if not all(sc_preds == predictions):
       import warnings; warnings.warn("SELF-CHECK MISMATCH iteration {iteration}")
9. Print metrics EXACTLY as:

print("METRICS_JSON_START")
print(json.dumps(metrics, indent=2, default=str))
print("METRICS_JSON_END")

metrics dict — include the actual best hyperparameters found:
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
            current_code_path.read_text(), stderr, stdout, attempt,
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
    """Determine strategy from counters, generate code, write it, return metadata dict."""
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
    script_path = GENERATED_CODE_DIR / script_name
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
