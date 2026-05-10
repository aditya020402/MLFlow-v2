"""
LangGraph-based agentic pipeline (CLI / pip-package edition).

Changes vs the backend version:
  - No HTTP / SSE streaming — events are printed to stdout instead.
  - No SqliteSaver checkpointer — runs in-process without persistence.
  - Human approval uses blocking input() instead of interrupt_before.
  - All path references routed through ada.paths (_paths).
"""

import json
import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional, TypedDict, Literal

from langgraph.graph import StateGraph, END

from ada import paths as _paths
from ada.agents import data_understanding, task_profiler, data_analyst, ml_engineer, evaluator, optimizer
from ada.agents.token_tracker import set_session as _set_session, get as _get_token_usage
from ada.tools.runner import run_script_with_retry, parse_metrics_from_output, get_primary_score

logger = logging.getLogger(__name__)

# Suppress LangGraph's pending deprecation warning about JsonPlusSerializer.allowed_objects
# — the parameter is internal to SqliteSaver and cannot be passed by callers.
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=DeprecationWarning,
)

MAX_RETRIES            = int(os.environ.get("MAX_RETRIES", "10"))
MAX_OPTIMIZATION_LOOPS = int(os.environ.get("MAX_OPTIMIZATION_LOOPS", "3"))
MAX_TUNE_ITERATIONS    = int(os.environ.get("MAX_TUNE_ITERATIONS", "2"))


# ─────────────────────────────── State ────────────────────────

class PipelineState(TypedDict):
    session_id: str
    dataset_path: str
    task_type: str
    target_column: Optional[str]

    understanding_code: str
    understanding_output: str
    understanding_error: str
    understanding_retries: int

    profiling_code: str
    profiling_output: str
    profiling_error: str
    profiling_retries: int

    analysis_code: str
    analysis_output: str
    analysis_error: str
    analysis_retries: int
    analysis_data: dict

    human_approved: bool
    human_feedback: str

    ml_code: str
    ml_output: str
    ml_error: str
    ml_retries: int
    algorithm_info: dict

    optimization_iteration: int
    optimization_history: list
    best_model: dict
    tried_algorithms: list
    models_tried: int
    current_algo_tune_count: int
    current_algorithm: str

    opt_script_name: str
    opt_algorithm: str
    opt_strategy: str
    opt_code: str
    opt_output: str
    opt_error: str
    last_opt_working_code: str

    evaluation: dict
    events: list
    validation_csv_path: str
    validation_results: dict
    token_usage: dict
    status: str
    final_model_filename: str


# ─────────────────────────── Helpers ──────────────────────────

_STEP_ICONS = {
    "agent_thinking":      "...",
    "code_generated":      "-->",
    "executing":           ">>>",
    "execution_success":   " OK",
    "execution_error":     "ERR",
    "awaiting_approval":   "???",
    "approved":            " OK",
    "algorithm_selected":  "ALG",
    "algorithm_rationale": "WHY",
    "best_model_updated":  "NEW",
    "iteration_tracked":   "TRK",
    "optimization_complete": "OPT",
    "evaluation_done":     "EVL",
    "validation_results":  "VAL",
    "validation_error":    "VAL",
    "token_usage":         "TOK",
    "completed":           "DONE",
}


def _print_rationale(algo_info: dict, strategy: str = "baseline", iteration: int = 0) -> None:
    reason = algo_info.get("reason", "")
    hyperparams = algo_info.get("hyperparameters", {})
    if reason:
        print(f"[WHY] {reason}", flush=True)
    if hyperparams:
        hp_str = ", ".join(f"{k}={v}" for k, v in list(hyperparams.items())[:6])
        print(f"[WHY] Hyperparameters: {hp_str}", flush=True)


def _emit(state: PipelineState, event_type: str, message: str, data: Any = None) -> list:
    events = list(state.get("events", []))
    events.append({"type": event_type, "message": message, "data": data})
    icon = _STEP_ICONS.get(event_type, "---")
    print(f"[{icon}] {message}", flush=True)
    logger.debug("[EVENT] %s: %s", event_type, message)
    return events


def _primary_metric_key(task_type: str) -> str:
    if "classif" in task_type:
        return "f1"
    if "regress" in task_type:
        return "r2_score"
    return "silhouette_score"


_METRIC_META_KEYS = {
    "algorithm", "iteration", "task_type", "model_path",
    "train_samples", "test_samples", "hyperparameters", "strategy",
}


def _format_metrics(metrics: dict, primary_key: str) -> str:
    parts = []
    if primary_key in metrics and isinstance(metrics[primary_key], (int, float)):
        parts.append(f"{primary_key}={metrics[primary_key]:.4f}")
    for k, v in metrics.items():
        if k == primary_key or k in _METRIC_META_KEYS:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, int):
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else f"{primary_key}=n/a"


def _try_parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return {}


# ─────────────────────────── Nodes ────────────────────────────

def node_data_understanding(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "General Understanding Agent computing universal dataset metrics...")

    result = data_understanding.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column"),
    )

    events = _emit({"events": events}, "code_generated", "step1a_general.py written.", data=result["code"])
    return {**state, "understanding_code": result["code"], "events": events, "understanding_retries": 0}


def node_execute_understanding(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "executing", "Running step1a_general.py...")

    script_path = _paths.GENERATED_CODE_DIR / "step1a_general.py"
    _session_log = _paths.OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step1a_general.py",
        fix_callback=data_understanding.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "General understanding complete.")
        understanding_parsed = _try_parse_json(result.stdout)
        return {**state, "understanding_output": result.stdout, "understanding_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"General understanding failed after {MAX_RETRIES} retries.")
    return {**state, "understanding_output": "", "understanding_error": result.stderr, "status": "failed", "events": events}


def node_task_profiling(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking",
                   f"Task Profiling Agent computing {state['task_type']}-specific metrics...")

    result = task_profiler.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column"),
        general_output=state.get("understanding_output", ""),
    )

    events = _emit({"events": events}, "code_generated", "step1b_profiling.py written.")
    return {**state, "profiling_code": result["code"], "events": events, "profiling_retries": 0}


def node_execute_profiling(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "executing", "Running step1b_profiling.py...")

    script_path = _paths.GENERATED_CODE_DIR / "step1b_profiling.py"
    _session_log = _paths.OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step1b_profiling.py",
        fix_callback=task_profiler.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "Task profiling complete.")
        return {**state, "profiling_output": result.stdout, "profiling_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"Task profiling failed after {MAX_RETRIES} retries.")
    return {**state, "profiling_output": "", "profiling_error": result.stderr, "status": "failed", "events": events}


def node_data_analysis(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "Data Analyst Agent generating cleaning and analysis code...")

    result = data_analyst.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column"),
        general_output=state.get("understanding_output", ""),
        profiling_output=state.get("profiling_output", ""),
        session_id=state.get("session_id", ""),
    )

    events = _emit({"events": events}, "code_generated", "step2_analysis.py written.")
    return {**state, "analysis_code": result["code"], "events": events, "analysis_retries": 0}


def node_execute_analysis(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "executing", "Running step2_analysis.py...")

    script_path = _paths.GENERATED_CODE_DIR / "step2_analysis.py"
    _session_log = _paths.OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step2_analysis.py",
        fix_callback=data_analyst.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "Data analysis complete.")
        analysis_data = _try_parse_json(result.stdout)

        cleaned_path = state["dataset_path"]
        for line in result.stdout.splitlines():
            if line.startswith("CLEANED_CSV:"):
                candidate = line[len("CLEANED_CSV:"):].strip()
                if candidate and Path(candidate).exists():
                    cleaned_path = candidate
                break

        return {
            **state,
            "dataset_path":    cleaned_path,
            "analysis_output": result.stdout,
            "analysis_data":   analysis_data or {},
            "analysis_error":  "",
            "status":          "awaiting_approval",
            "events":          events,
        }

    events = _emit({"events": events}, "execution_error", f"Data analysis failed after {MAX_RETRIES} retries.")
    return {**state, "analysis_output": "", "analysis_error": result.stderr, "status": "failed", "events": events}


def node_human_approval(state: PipelineState) -> PipelineState:
    """Blocking CLI approval gate — prompts the user, then always proceeds."""
    analysis_data = state.get("analysis_data", {})

    print("\n" + "=" * 60)
    print("DATA ANALYSIS COMPLETE — Review before ML modeling")
    print("=" * 60)

    if analysis_data:
        shape_orig = analysis_data.get("original_shape", "?")
        shape_clean = analysis_data.get("cleaned_shape", "?")
        print(f"  Original shape : {shape_orig}")
        print(f"  Cleaned shape  : {shape_clean}")

        recs = analysis_data.get("recommendations", [])
        if isinstance(recs, str):
            recs = [recs] if recs else []
        if recs:
            print("\n  Recommendations:")
            for r in recs[:5]:
                print(f"    - {r}")

        ml_recs = analysis_data.get("ml_model_recommendations", [])
        if isinstance(ml_recs, str):
            ml_recs = [ml_recs] if ml_recs else []
        elif isinstance(ml_recs, dict):
            ml_recs = ml_recs.get("algorithm_hints", [])
        if ml_recs:
            print("\n  Suggested models:")
            for m in ml_recs[:3]:
                print(f"    - {m}")

    print("\n  Press Enter to proceed to ML modeling.")
    print("  You may also type optional feedback/instructions and press Enter.")
    print("-" * 60)

    try:
        feedback = input("  Your feedback (or just Enter): ").strip()
    except (EOFError, KeyboardInterrupt):
        feedback = ""

    print("=" * 60 + "\n")

    events = _emit(state, "approved",
                   "User approved. Proceeding to ML Engineering." +
                   (f' Feedback: "{feedback[:120]}"' if feedback else ""))
    return {
        **state,
        "human_approved": True,
        "human_feedback": feedback,
        "status": "running",
        "events": events,
    }


def node_ml_engineering(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "ML Engineer Agent selecting algorithm and generating baseline code...")

    current_tried = list(state.get("tried_algorithms", []))

    _understanding = (
        state.get("understanding_output", "") +
        "\n\n=== TASK-SPECIFIC PROFILE ===\n" +
        state.get("profiling_output", "")
    )
    result = ml_engineer.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column", ""),
        understanding_output=_understanding,
        analysis_output=state["analysis_output"],
        analysis_data=state.get("analysis_data", {}),
        human_feedback=state.get("human_feedback", ""),
        tried_algorithms=current_tried,
        session_id=state.get("session_id", ""),
    )

    algo_info = result["algorithm_info"]
    chosen_algo = algo_info["algorithm"]
    updated_tried = current_tried + [chosen_algo]

    events = _emit({"events": events}, "algorithm_selected",
                   f"Baseline algorithm: {chosen_algo}", data=algo_info)
    _print_rationale(algo_info, strategy="baseline")
    events = _emit({"events": events}, "code_generated", "step3_ml.py (baseline) written.")
    return {
        **state,
        "ml_code": result["code"],
        "algorithm_info": algo_info,
        "tried_algorithms": updated_tried,
        "current_algorithm": chosen_algo,
        "events": events,
        "ml_retries": 0,
    }


def node_execute_ml(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "executing", "Running step3_ml.py — baseline...")

    script_path = _paths.GENERATED_CODE_DIR / "step3_ml.py"
    _session_log = _paths.OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step3_ml.py",
        fix_callback=ml_engineer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        working_code = (_paths.GENERATED_CODE_DIR / "step3_ml.py").read_text(encoding="utf-8")
        events = _emit({"events": events}, "execution_success", "Baseline model trained.")
        return {**state, "ml_output": result.stdout, "ml_error": "",
                "last_opt_working_code": working_code, "events": events}

    events = _emit({"events": events}, "execution_error", f"Baseline failed after {MAX_RETRIES} retries.")
    return {**state, "ml_output": "", "ml_error": result.stderr, "status": "failed", "events": events}


def node_track_metrics(state: PipelineState) -> PipelineState:
    opt_iter = state.get("optimization_iteration", 0)

    if opt_iter == 0:
        raw_output = state.get("ml_output", "")
        algo = state.get("algorithm_info", {}).get("algorithm", "unknown")
    else:
        raw_output = state.get("opt_output", "")
        algo = state.get("opt_algorithm", "unknown")

    metrics = parse_metrics_from_output(raw_output)
    primary_score = get_primary_score(metrics)
    primary_key = _primary_metric_key(state["task_type"])

    strategy = "baseline" if opt_iter == 0 else state.get("opt_strategy", "new_algorithm")

    history_entry = {
        "iteration": opt_iter,
        "strategy": strategy,
        "algorithm": algo,
        "metrics": metrics,
        "primary_score": primary_score,
        "primary_metric": primary_key,
    }

    history = list(state.get("optimization_history", []))
    history.append(history_entry)

    best = dict(state.get("best_model", {}))
    metrics_str = _format_metrics(metrics, primary_key)
    if primary_score > best.get("primary_score", -1.0):
        best = {
            "model_path": metrics.get("model_path", str(_paths.OUTPUTS_DIR / f"model_iter{opt_iter}.pkl")),
            "algorithm": algo,
            "primary_score": primary_score,
            "primary_metric": primary_key,
            "metrics": metrics,
            "iteration": opt_iter,
        }
        events = _emit(state, "best_model_updated",
                       f"New best: {algo} (iter {opt_iter}) — {metrics_str}", data=best)
    else:
        events = _emit(state, "iteration_tracked",
                       f"Iter {opt_iter}: {algo} — {metrics_str} "
                       f"(best={best.get('primary_score', 0):.4f})")

    return {**state, "optimization_history": history, "best_model": best, "events": events}


def node_optimizer(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    next_iter = state.get("optimization_iteration", 0) + 1
    current_tune = state.get("current_algo_tune_count", 0)
    current_models = state.get("models_tried", 0)

    events = _emit(
        state, "agent_thinking",
        f"Optimizer — model {current_models}/{MAX_OPTIMIZATION_LOOPS}, "
        f"tune round {current_tune}/{MAX_TUNE_ITERATIONS} (pass {next_iter})...",
    )

    current_tried = list(state.get("tried_algorithms", []))
    _understanding = (
        state.get("understanding_output", "") +
        "\n\n=== TASK-SPECIFIC PROFILE ===\n" +
        state.get("profiling_output", "")
    )
    result = optimizer.run(
        dataset_path=state["dataset_path"],
        task_type=state["task_type"],
        target_column=state.get("target_column", ""),
        iteration=next_iter,
        optimization_history=state.get("optimization_history", []),
        understanding_output=_understanding,
        best_model=state.get("best_model", {}),
        current_algo_tune_count=current_tune,
        models_tried=current_models,
        human_feedback=state.get("human_feedback", ""),
        tried_algorithms=current_tried,
        last_working_code=state.get("last_opt_working_code", ""),
        current_algorithm=state.get("current_algorithm", ""),
        session_id=state.get("session_id", ""),
        analysis_data=state.get("analysis_data", {}),
    )

    algo_info = result["algorithm_info"]
    chosen_algo = algo_info["algorithm"]
    chosen_strategy = result.get("strategy", "new_algorithm")

    if chosen_strategy == "tune_best":
        new_tune_count   = current_tune + 1
        new_models_tried = current_models
        updated_tried    = current_tried
        new_current_algo = state.get("current_algorithm", chosen_algo)
    else:
        new_tune_count   = 0
        new_models_tried = current_models + 1
        updated_tried    = current_tried + [chosen_algo]
        new_current_algo = chosen_algo

    events = _emit({"events": events}, "algorithm_selected",
                   f"Pass {next_iter} [{chosen_strategy}]: {chosen_algo} "
                   f"(model {new_models_tried}/{MAX_OPTIMIZATION_LOOPS}, "
                   f"tune {new_tune_count}/{MAX_TUNE_ITERATIONS})",
                   data=algo_info)
    _print_rationale(algo_info, strategy=chosen_strategy, iteration=next_iter)
    events = _emit({"events": events}, "code_generated",
                   f"step3_ml_iter{next_iter}.py written — [{chosen_strategy}] {chosen_algo}")

    return {
        **state,
        "optimization_iteration": next_iter,
        "models_tried": new_models_tried,
        "current_algo_tune_count": new_tune_count,
        "tried_algorithms": updated_tried,
        "current_algorithm": new_current_algo,
        "opt_script_name": result["script_name"],
        "opt_algorithm": chosen_algo,
        "opt_strategy": chosen_strategy,
        "opt_code": result["code"],
        "events": events,
    }


def node_execute_optimization(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    script_name = state.get("opt_script_name", "")
    iteration = state.get("optimization_iteration", 1)

    if not script_name:
        err = f"opt_script_name is empty for iteration {iteration}."
        events = _emit(state, "execution_error", err)
        return {**state, "opt_output": "", "opt_error": err, "status": "failed", "events": events}

    events = _emit(state, "executing", f"Running {script_name} (iteration {iteration})...")

    script_path = _paths.GENERATED_CODE_DIR / script_name
    _session_log = _paths.OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        script_name,
        fix_callback=optimizer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        working_code = (_paths.GENERATED_CODE_DIR / script_name).read_text(encoding="utf-8") if script_name else ""
        events = _emit({"events": events}, "execution_success", f"Iteration {iteration} succeeded.")
        return {**state, "opt_output": result.stdout, "opt_error": "",
                "last_opt_working_code": working_code, "events": events}

    events = _emit({"events": events}, "execution_error",
                   f"Iteration {iteration} failed after {MAX_RETRIES} retries.")
    return {**state, "opt_output": "", "opt_error": result.stderr, "status": "failed", "events": events}


def _safe_name(s: str) -> str:
    """Sanitize a string for use in a filename (keep alphanumerics and underscores)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")[:40]


def node_finalize_best(state: PipelineState) -> PipelineState:
    best = state.get("best_model", {})
    history = state.get("optimization_history", [])

    events = _emit(state, "agent_thinking", "Finalizing best model across all iterations...")

    session_out = _paths.OUTPUTS_DIR / (state.get("session_id") or "default")
    session_out.mkdir(parents=True, exist_ok=True)

    best_path = best.get("model_path", "")

    # Build a descriptive filename: best_{Algorithm}_{metric}_{score:.4f}.pkl
    algo_safe    = _safe_name(best.get("algorithm", "model"))
    metric_safe  = _safe_name(best.get("primary_metric", "score"))
    score_val    = best.get("primary_score", 0.0) or 0.0
    model_filename = f"best_{algo_safe}_{metric_safe}_{score_val:.4f}.pkl"
    final_path     = str(session_out / model_filename)

    if best_path and Path(best_path).exists():
        shutil.copy2(best_path, final_path)
        # Also copy the companion _features.pkl with matching name
        features_src = Path(best_path.replace(".pkl", "_features.pkl"))
        if features_src.exists():
            shutil.copy2(features_src, str(session_out / model_filename.replace(".pkl", "_features.pkl")))
        logger.info("Copied best model %s -> %s", best_path, final_path)
    else:
        logger.warning("Best model path not found: %s", best_path)

    history_path = session_out / "optimization_history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    events = _emit(
        {"events": events}, "optimization_complete",
        f"Optimization done. Best: {best.get('algorithm')} "
        f"({best.get('primary_metric')}={best.get('primary_score', 0):.4f}) → {model_filename}",
        data={"best_model": best, "history": history, "model_filename": model_filename},
    )
    return {**state, "final_model_filename": model_filename, "events": events}


def node_evaluation(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "Evaluation Agent analyzing best model performance...")

    best = state.get("best_model", {})
    algo = best.get("algorithm", state.get("algorithm_info", {}).get("algorithm", "unknown"))
    metrics_output = json.dumps(best.get("metrics", {})) or state.get("ml_output", "")

    evaluation = evaluator.evaluate(
        task_type=state["task_type"],
        algorithm=algo,
        metrics_output=metrics_output,
        understanding_output=state.get("understanding_output", ""),
        primary_metric=best.get("primary_metric", _primary_metric_key(state["task_type"])),
    )

    verdict = evaluation.get("verdict", "pass")
    score = evaluation.get("score", 0.0)

    events = _emit(
        {"events": events}, "evaluation_done",
        f"Evaluation complete. Verdict: {verdict.upper()} | Score: {score:.3f}",
        data={**evaluation, "best_model": best, "optimization_history": state.get("optimization_history", [])},
    )

    session_id = state.get("session_id", "default")

    # Optional 5% held-out validation
    val_csv = state.get("validation_csv_path", "")
    val_result = {}
    if val_csv and Path(val_csv).exists():
        events = _emit({"events": events}, "agent_thinking",
                       "Running validation agent on held-out 5% data...")
        try:
            from ada.agents import testing_agent as _ta
            _model_file = state.get("final_model_filename") or "model.pkl"
            model_path = str(_paths.OUTPUTS_DIR / session_id / _model_file)
            training_code = state.get("last_opt_working_code", "")
            val_result = _ta.run(
                test_csv_path=val_csv,
                model_path=model_path,
                task_type=state["task_type"],
                target_column=state.get("target_column"),
                training_code=training_code,
                session_id=session_id,   # same session so tokens are tracked together
            )
            if "error" not in val_result:
                val_metrics = val_result.get("metrics", {})
                events = _emit({"events": events}, "validation_results",
                               f"Held-out validation: {_format_metrics(val_metrics, _primary_metric_key(state['task_type']))}",
                               data=val_result)
            else:
                _err_msg = val_result.get("error", "unknown")
                val_result = {}
                events = _emit({"events": events}, "validation_error",
                               f"Validation failed: {_err_msg}")
        except Exception as exc:
            val_result = {}
            logger.error("Validation agent error: %s", exc, exc_info=True)
            events = _emit({"events": events}, "validation_error", f"Validation error: {exc}")

    # Collect token usage AFTER validation so testing_agent tokens are included
    token_usage = _get_token_usage(session_id)
    if token_usage:
        events = _emit({"events": events}, "token_usage", "Token usage summary ready.", data=token_usage)

    events = _emit({"events": events}, "completed", "Pipeline finished successfully!")
    return {
        **state,
        "evaluation": evaluation,
        "token_usage": token_usage,
        "validation_results": val_result,
        "status": "completed",
        "events": events,
    }


# ─────────────────────────── Edges ────────────────────────────

def edge_after_understanding_exec(state: PipelineState) -> Literal["task_profiling", "__end__"]:
    return "__end__" if state.get("understanding_error") else "task_profiling"


def edge_after_profiling_exec(state: PipelineState) -> Literal["data_analysis", "__end__"]:
    return "__end__" if state.get("profiling_error") else "data_analysis"


def edge_after_analysis_exec(state: PipelineState) -> Literal["human_approval", "__end__"]:
    return "__end__" if state.get("analysis_error") else "human_approval"


def edge_after_ml_exec(state: PipelineState) -> Literal["track_metrics", "__end__"]:
    return "__end__" if state.get("ml_error") else "track_metrics"


def edge_optimization_loop(state: PipelineState) -> Literal["optimizer", "finalize_best"]:
    tune_done   = state.get("current_algo_tune_count", 0) >= MAX_TUNE_ITERATIONS
    models_done = state.get("models_tried", 0) >= MAX_OPTIMIZATION_LOOPS
    if tune_done and models_done:
        return "finalize_best"
    return "optimizer"


def edge_after_opt_exec(state: PipelineState) -> Literal["track_metrics", "finalize_best"]:
    # On failure, skip to finalize_best so the best model found so far is still saved
    return "finalize_best" if state.get("opt_error") else "track_metrics"


def edge_after_track(state: PipelineState) -> Literal["optimization_loop", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "optimization_loop"


# ─────────────────────────── Graph ────────────────────────────

def build_graph():
    workflow = StateGraph(PipelineState)

    workflow.add_node("data_understanding",    node_data_understanding)
    workflow.add_node("execute_understanding", node_execute_understanding)
    workflow.add_node("task_profiling",        node_task_profiling)
    workflow.add_node("execute_profiling",     node_execute_profiling)
    workflow.add_node("data_analysis",         node_data_analysis)
    workflow.add_node("execute_analysis",      node_execute_analysis)
    workflow.add_node("human_approval",        node_human_approval)
    workflow.add_node("ml_engineering",        node_ml_engineering)
    workflow.add_node("execute_ml",            node_execute_ml)
    workflow.add_node("track_metrics",         node_track_metrics)
    workflow.add_node("optimization_loop",     lambda s: s)
    workflow.add_node("optimizer",             node_optimizer)
    workflow.add_node("execute_optimization",  node_execute_optimization)
    workflow.add_node("finalize_best",         node_finalize_best)
    workflow.add_node("evaluation",            node_evaluation)

    workflow.set_entry_point("data_understanding")

    workflow.add_edge("data_understanding", "execute_understanding")
    workflow.add_conditional_edges(
        "execute_understanding",
        edge_after_understanding_exec,
        {"task_profiling": "task_profiling", "__end__": END},
    )
    workflow.add_edge("task_profiling", "execute_profiling")
    workflow.add_conditional_edges(
        "execute_profiling",
        edge_after_profiling_exec,
        {"data_analysis": "data_analysis", "__end__": END},
    )
    workflow.add_edge("data_analysis", "execute_analysis")
    workflow.add_conditional_edges(
        "execute_analysis",
        edge_after_analysis_exec,
        {"human_approval": "human_approval", "__end__": END},
    )
    # human_approval always proceeds after blocking input() — no loop back needed
    workflow.add_edge("human_approval", "ml_engineering")
    workflow.add_edge("ml_engineering", "execute_ml")
    workflow.add_conditional_edges(
        "execute_ml",
        edge_after_ml_exec,
        {"track_metrics": "track_metrics", "__end__": END},
    )
    workflow.add_conditional_edges(
        "track_metrics",
        edge_after_track,
        {"optimization_loop": "optimization_loop", "__end__": END},
    )
    workflow.add_conditional_edges(
        "optimization_loop",
        edge_optimization_loop,
        {"optimizer": "optimizer", "finalize_best": "finalize_best"},
    )
    workflow.add_edge("optimizer", "execute_optimization")
    workflow.add_conditional_edges(
        "execute_optimization",
        edge_after_opt_exec,
        {"track_metrics": "track_metrics", "finalize_best": "finalize_best"},
    )
    workflow.add_edge("finalize_best", "evaluation")
    workflow.add_edge("evaluation", END)

    # No checkpointer, no interrupt_before — pure in-process run
    return workflow.compile()


def create_initial_state(
    dataset_path: str,
    task_type: str,
    target_column: str = None,
    session_id: str = "",
    validation_csv_path: str = "",
) -> PipelineState:
    return PipelineState(
        session_id=session_id,
        dataset_path=dataset_path,
        task_type=task_type,
        target_column=target_column,
        understanding_code="",
        understanding_output="",
        understanding_error="",
        understanding_retries=0,
        profiling_code="",
        profiling_output="",
        profiling_error="",
        profiling_retries=0,
        analysis_code="",
        analysis_output="",
        analysis_error="",
        analysis_retries=0,
        analysis_data={},
        human_approved=False,
        human_feedback="",
        ml_code="",
        ml_output="",
        ml_error="",
        ml_retries=0,
        algorithm_info={},
        optimization_iteration=0,
        optimization_history=[],
        best_model={},
        tried_algorithms=[],
        models_tried=0,
        current_algo_tune_count=0,
        current_algorithm="",
        opt_script_name="",
        opt_algorithm="",
        opt_strategy="",
        opt_code="",
        opt_output="",
        opt_error="",
        last_opt_working_code="",
        evaluation={},
        events=[],
        token_usage={},
        status="running",
        validation_csv_path=validation_csv_path,
        validation_results={},
        final_model_filename="",
    )
