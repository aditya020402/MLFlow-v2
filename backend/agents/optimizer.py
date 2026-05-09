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

from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

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
- ALWAYS use train_test_split(X, y, test_size=0.2, random_state=42).
- Fit ALL preprocessing (scalers, encoders) on TRAIN only — transform both sets.
- Compute ALL metrics on TEST SET ONLY.
- Wrap the final metrics JSON in METRICS_JSON_START / METRICS_JSON_END markers.
- Save the model with pickle to the exact path provided.
- Include self-check: reload model, re-predict on test, warn if metrics diverge.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks."""

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
        return '"r2_score": <float>, "rmse": <float>, "mae": <float>'
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
        prompt = f"""You are an ML hyperparameter tuning expert.

Current model to tune: {tune_algo}
  Current score ({best_metric}): {tune_score:.4f}
  Current hyperparameters: {json.dumps(tune_hyperparams, indent=2)}
  Tune round: {current_algo_tune_count + 1} of {MAX_TUNE_ITERATIONS}
  Overall best so far: {best_algo} at {best_score:.4f} {best_metric}

Task type: {task_type}
Optimization history:
{history_summary}
{feedback_section}

Provide the hyperparameter search space for RandomizedSearchCV to improve {tune_algo}.
Focus on ranges most likely to push {best_metric} beyond {tune_score:.4f}.

For the `reason` field write 2-3 sentences explaining:
1. What the current results show and which hyperparameter dimensions have the most room to improve.
2. Why you chose these specific search ranges.
3. What score improvement you expect.

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
        prompt = f"""You are an ML algorithm selection expert.

The current model ({tune_algo}) has been tuned {current_algo_tune_count} time(s) — budget exhausted.
Best score so far: {best_score:.4f} ({best_metric}) achieved by {best_algo}

Task type: {task_type}
Already tried (DO NOT pick these): {already_tried}

Optimization history:
{history_summary}

Data characteristics:
{understanding_output[:1500]}
{feedback_section}

Pick the NEXT ML algorithm most likely to beat {best_score:.4f} ({best_metric}).
Provide both `hyperparameters` (initial values) and `search_space` (for a quick RandomizedSearchCV).

For the `reason` field write 2-3 sentences explaining:
1. Why {best_algo} has reached its limit and why a new approach is needed.
2. Which specific data or result characteristics make this new algorithm a better fit.
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

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        result["strategy"] = strategy   # always override — LLM does not decide this

        # For Path B, guarantee the LLM didn't hallucinate a duplicate algorithm.
        if strategy == "new_algorithm":
            chosen = result.get("algorithm", "")
            if chosen in already_tried:
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
    stratify = "stratify=y, " if "classif" in task_type else ""
    scale = _scale_needed(algo)
    scoring = _scoring_metric(task_type)
    x_eval = "X_test_scaled" if scale else "X_test"

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

    # ── PATH A with working code — template adaptation (avoids repeating past bugs) ──
    if strat_type == "tune_best" and last_working_code.strip():
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
        code = _strip_markdown(response.choices[0].message.content.strip())
        logger.info("Optimizer [iter %d] tune_best template adapted for %s", iteration, algo)
        return code

    # ── PATH A without working code OR PATH B — generate from scratch ──────
    if strat_type == "tune_best":
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
       search.fit(X_train{"_scaled" if scale else ""}, y_train)
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
       search.fit(X_train{"_scaled" if scale else ""}, y_train)
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

Requirements (follow EXACTLY):
1. pandas read_csv to load dataset
2. Preprocessing (fit on train only):
   - Drop cols with >50% missing
   - Fill numeric NaN (median for skewed, mean for symmetric)
   - LabelEncode all object/category cols
   {"- Separate X and y (target='" + target_column + "')" if target_column else "- X = all columns"}
3. train_test_split(X, y, test_size=0.2, random_state=42, {stratify})
4. {"StandardScaler: fit X_train, transform X_train + X_test → X_train_scaled, X_test_scaled" if scale else "No scaling needed"}
{tuning_section}
6. Compute on TEST SET ONLY using best_model_instance.predict({x_eval}):
   {_metrics_fields(task_type)}
7. pickle.dump(best_model_instance, open("{model_path}", "wb"))
8. SELF-CHECK (data leakage guard):
   sc_model = pickle.load(open("{model_path}", "rb"))
   sc_preds = sc_model.predict({x_eval})
   # Compare sc_preds to original predictions — warn if they differ
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

    code = _strip_markdown(response.choices[0].message.content.strip())
    logger.info("Optimizer [iter %d] strategy=%s algorithm=%s", iteration, strat_type, algo)
    return code


def fix_optimization_code(
    current_code: str, stderr: str, stdout: str, attempt: int
) -> str:
    logger.info("Optimizer fixing code (attempt %d)...", attempt)

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
            {"role": "user", "content": prompt},
        ],
    )
    return _strip_markdown(response.choices[0].message.content.strip())


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        return fix_optimization_code(
            current_code_path.read_text(), stderr, stdout, attempt
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
) -> dict:
    """Determine strategy from counters, generate code, write it, return metadata dict."""
    strategy = select_next_strategy(
        task_type, iteration, optimization_history,
        understanding_output, best_model,
        current_algo_tune_count, models_tried,
        human_feedback,
        tried_algorithms=tried_algorithms or [],
        current_algorithm=current_algorithm,
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
    )

    script_name = f"step3_ml_iter{iteration}.py"
    script_path = GENERATED_CODE_DIR / script_name
    script_path.write_text(code)
    logger.info("Written: %s", script_path)

    return {
        "script_name": script_name,
        "script_path": str(script_path),
        "code": code,
        "algorithm_info": strategy,
        "strategy": strategy.get("strategy", "new_algorithm"),
        "fix_callback": make_fix_callback(script_path),
    }
