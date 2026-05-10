"""
LangGraph-based agentic pipeline with optimization loop.

Flow:
  data_understanding → execute_understanding
    → data_analysis → execute_analysis
    → [HUMAN APPROVAL with optional feedback]
    → ml_engineering → execute_ml
    → track_metrics                       ← parses metrics, updates best model
    → optimization_loop                   ← conditional: more iters OR finalize
        ↓ (iter < MAX_OPT_ITERATIONS)
      optimizer → execute_optimization → track_metrics (loop)
        ↓ (iter >= MAX_OPT_ITERATIONS)
      finalize_best → evaluation → END
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, TypedDict, Literal

# Suppress LangGraph's pending deprecation warning about JsonPlusSerializer.allowed_objects
# — the parameter is internal to SqliteSaver and cannot be passed by callers.
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=DeprecationWarning,
)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import data_understanding, task_profiler, data_analyst, ml_engineer, evaluator, optimizer
from agents.token_tracker import set_session as _set_session, get as _get_token_usage
from tools.runner import run_script_with_retry, parse_metrics_from_output, get_primary_score

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DB_PATH = Path(__file__).parent.parent / "outputs" / "checkpoints.db"

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "10"))
# MAX_OPTIMIZATION_LOOPS = number of distinct ML models tried after the baseline.
# MAX_TUNE_ITERATIONS    = hyperparameter tuning rounds per model (including baseline).
MAX_OPTIMIZATION_LOOPS = int(os.environ.get("MAX_OPTIMIZATION_LOOPS", "3"))
MAX_TUNE_ITERATIONS    = int(os.environ.get("MAX_TUNE_ITERATIONS", "2"))


# ─────────────────────── Persistent checkpointer ──────────────

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


# ─────────────────────────────── State ────────────────────────

class PipelineState(TypedDict):
    # ── Inputs ──
    session_id: str
    dataset_path: str       # updated to cleaned CSV path after data analysis
    task_type: str          # supervised_classification | supervised_regression | unsupervised
    target_column: Optional[str]

    # ── Step 1a — General Data Understanding ──
    understanding_code: str
    understanding_output: str   # JSON from step1a_general.py
    understanding_error: str
    understanding_retries: int

    # ── Step 1b — Task-Specific Profiling ──
    profiling_code: str
    profiling_output: str       # JSON from step1b_profiling.py
    profiling_error: str
    profiling_retries: int

    # ── Step 2 — Data Analysis ──
    analysis_code: str
    analysis_output: str
    analysis_error: str
    analysis_retries: int
    analysis_data: dict     # parsed JSON from analysis script — col types, encoding recs, imbalance, etc.

    # ── Human approval gate ──
    human_approved: bool
    human_feedback: str     # optional free-text feedback from user at approval gate

    # ── Step 3 — ML Engineering (baseline) ──
    ml_code: str
    ml_output: str
    ml_error: str
    ml_retries: int
    algorithm_info: dict

    # ── Optimization loop ──
    optimization_iteration: int     # increments every optimizer pass (used for script naming)
    optimization_history: list      # [{iteration, strategy, algorithm, metrics, primary_score}]
    best_model: dict                # {model_path, algorithm, primary_score, metrics, iteration}
    tried_algorithms: list          # flat list of every algorithm name selected (baseline + all opt iters)
    models_tried: int               # distinct ML algorithms tried after baseline (0 → MAX_OPTIMIZATION_LOOPS)
    current_algo_tune_count: int    # tuning rounds done for the current model (resets on model switch)
    current_algorithm: str          # algorithm we are currently exploring (set on new_algorithm; used by tune_best)

    # ── Optimizer scratch (must be in TypedDict so SQLite checkpointer persists them) ──
    opt_script_name: str            # e.g. step3_ml_iter1.py
    opt_algorithm: str              # algorithm name for current iteration
    opt_strategy: str               # "tune_best" | "new_algorithm" | "baseline"
    opt_code: str                   # generated code for current iteration
    opt_output: str                 # stdout from execute_optimization
    opt_error: str                  # stderr if failed
    last_opt_working_code: str      # final working script content (post-fixes) for current algorithm

    # ── Evaluation ──
    evaluation: dict

    # ── UI streaming events ──
    events: list

    # ── Validation (optional 5% hold-out) ──
    validation_csv_path: str  # path to held-out validation CSV, or ""
    validation_results: dict  # populated by node_evaluation after running testing_agent

    # ── Token usage ──
    token_usage: dict

    # ── Final status ──
    status: str     # running | awaiting_approval | completed | failed


# ─────────────────────────── Helpers ──────────────────────────

def _emit(state: PipelineState, event_type: str, message: str, data: Any = None) -> list:
    events = list(state.get("events", []))
    events.append({"type": event_type, "message": message, "data": data})
    logger.info("[EVENT] %s: %s", event_type, message)
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
    """Return a compact 'key=val, ...' string of all numeric metrics for console messages."""
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
    """Attempt to parse JSON from stdout; return empty dict on failure."""
    try:
        return json.loads(text)
    except Exception:
        # stdout may contain log lines before JSON; try to find the JSON block
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

    events = _emit(
        {"events": events}, "code_generated",
        "step1a_general.py written.", data=result["code"],
    )
    return {**state, "understanding_code": result["code"], "events": events, "understanding_retries": 0}


def node_execute_understanding(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    retries = state.get("understanding_retries", 0)
    events = _emit(state, "executing", f"Running step1a_general.py (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step1a_general.py"
    _session_log = OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step1a_general.py",
        fix_callback=data_understanding.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "General understanding complete.", data=result.stdout[:3000])
        understanding_parsed = _try_parse_json(result.stdout)
        if understanding_parsed:
            events = _emit({"events": events}, "understanding_data", "Understanding data ready.", data=understanding_parsed)
        return {**state, "understanding_output": result.stdout, "understanding_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"General understanding failed after {MAX_RETRIES} retries.", data=result.stderr)
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

    events = _emit({"events": events}, "code_generated", "step1b_profiling.py written.", data=result["code"])
    return {**state, "profiling_code": result["code"], "events": events, "profiling_retries": 0}


def node_execute_profiling(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    retries = state.get("profiling_retries", 0)
    events = _emit(state, "executing", f"Running step1b_profiling.py (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step1b_profiling.py"
    _session_log = OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step1b_profiling.py",
        fix_callback=task_profiler.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "Task profiling complete.", data=result.stdout[:3000])
        profiling_parsed = _try_parse_json(result.stdout)
        if profiling_parsed:
            events = _emit({"events": events}, "profiling_data", "Profiling data ready.", data=profiling_parsed)
        return {**state, "profiling_output": result.stdout, "profiling_error": "", "events": events}

    events = _emit({"events": events}, "execution_error", f"Task profiling failed after {MAX_RETRIES} retries.", data=result.stderr)
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

    events = _emit({"events": events}, "code_generated", "step2_analysis.py written.", data=result["code"])
    return {**state, "analysis_code": result["code"], "events": events, "analysis_retries": 0}


def node_execute_analysis(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    retries = state.get("analysis_retries", 0)
    events = _emit(state, "executing", f"Running step2_analysis.py (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step2_analysis.py"
    _session_log = OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step2_analysis.py",
        fix_callback=data_analyst.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        events = _emit({"events": events}, "execution_success", "step2 succeeded.", data=result.stdout[:3000])
        # Emit structured analysis data for the UI approval panel
        analysis_data = _try_parse_json(result.stdout)
        if analysis_data:
            events = _emit({"events": events}, "analysis_data", "Analysis results ready.", data=analysis_data)
        events = _emit({"events": events}, "awaiting_approval",
                       "Analysis complete. Awaiting human approval to proceed to modeling.")

        # Extract cleaned CSV path emitted by the analysis script
        cleaned_path = state["dataset_path"]
        for line in result.stdout.splitlines():
            if line.startswith("CLEANED_CSV:"):
                candidate = line[len("CLEANED_CSV:"):].strip()
                if candidate and Path(candidate).exists():
                    cleaned_path = candidate
                    logger.info("Using cleaned dataset: %s", cleaned_path)
                break

        return {
            **state,
            "dataset_path":   cleaned_path,
            "analysis_output": result.stdout,
            "analysis_data":   analysis_data or {},
            "analysis_error":  "",
            "status":          "awaiting_approval",
            "events":          events,
        }

    events = _emit({"events": events}, "execution_error", f"step2 failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "analysis_output": "", "analysis_error": result.stderr, "status": "failed", "events": events}


def node_human_approval(state: PipelineState) -> PipelineState:
    if state.get("human_approved"):
        feedback = state.get("human_feedback", "")
        msg = "Human approved. Proceeding to ML Engineering."
        if feedback:
            msg += f" Feedback: \"{feedback[:120]}\""
        events = _emit(state, "approved", msg)
        return {**state, "status": "running", "events": events}
    return {**state, "status": "awaiting_approval"}


def node_ml_engineering(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "ML Engineer Agent is selecting algorithm and generating baseline code...")

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
    events = _emit({"events": events}, "algorithm_rationale",
                   algo_info.get("reason", "No rationale provided."),
                   data={"algorithm": chosen_algo, "strategy": "baseline",
                         "hyperparameters": algo_info.get("hyperparameters", {})})
    events = _emit({"events": events}, "code_generated", "step3_ml.py (iteration 0) written.", data=result["code"])
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
    retries = state.get("ml_retries", 0)
    events = _emit(state, "executing", f"Running step3_ml.py — baseline (attempt {retries + 1})...")

    script_path = GENERATED_CODE_DIR / "step3_ml.py"
    _session_log = OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        "step3_ml.py",
        fix_callback=ml_engineer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        # Read the final file content — fix_callback may have rewritten it, so disk is authoritative.
        working_code = (GENERATED_CODE_DIR / "step3_ml.py").read_text()
        events = _emit({"events": events}, "execution_success", "Baseline model trained.", data=result.stdout[:3000])
        return {**state, "ml_output": result.stdout, "ml_error": "",
                "last_opt_working_code": working_code, "events": events}

    events = _emit({"events": events}, "execution_error", f"Baseline failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "ml_output": "", "ml_error": result.stderr, "status": "failed", "events": events}


def node_track_metrics(state: PipelineState) -> PipelineState:
    """Parse metrics from the most recently executed ML output, update best_model."""
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
            "model_path": metrics.get("model_path", str(OUTPUTS_DIR / f"model_iter{opt_iter}.pkl")),
            "algorithm": algo,
            "primary_score": primary_score,
            "primary_metric": primary_key,
            "metrics": metrics,
            "iteration": opt_iter,
        }
        events = _emit(state, "best_model_updated",
                       f"New best: {algo} (iter {opt_iter}) — {metrics_str}",
                       data=best)
    else:
        events = _emit(state, "iteration_tracked",
                       f"Iter {opt_iter}: {algo} — {metrics_str} "
                       f"(best={best.get('primary_score', 0):.4f})")

    return {
        **state,
        "optimization_history": history,
        "best_model": best,
        "events": events,
    }


def node_optimizer(state: PipelineState) -> PipelineState:
    """Generate code for the next optimization pass.

    Strategy is determined by the nested state machine:
      - current_algo_tune_count < MAX_TUNE_ITERATIONS  →  tune_best (Path A)
      - tune count exhausted, models_tried < MAX_OPTIMIZATION_LOOPS  →  new_algorithm (Path B)
    The LLM is only asked for *what* to do (search space / which new algo), not *whether* to tune.
    """
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

    # Update the nested counters based on what was decided.
    if chosen_strategy == "tune_best":
        new_tune_count     = current_tune + 1
        new_models_tried   = current_models
        updated_tried      = current_tried          # tuning same model — no new entry
        new_current_algo   = state.get("current_algorithm", chosen_algo)
    else:  # new_algorithm
        new_tune_count     = 0
        new_models_tried   = current_models + 1
        updated_tried      = current_tried + [chosen_algo]
        new_current_algo   = chosen_algo            # switch focus to the newly selected algo

    events = _emit({"events": events}, "algorithm_selected",
                   f"Pass {next_iter} [{chosen_strategy}]: {chosen_algo} "
                   f"(model {new_models_tried}/{MAX_OPTIMIZATION_LOOPS}, "
                   f"tune {new_tune_count}/{MAX_TUNE_ITERATIONS})",
                   data=algo_info)
    events = _emit({"events": events}, "algorithm_rationale",
                   algo_info.get("reason", "No rationale provided."),
                   data={"algorithm": chosen_algo, "strategy": chosen_strategy,
                         "iteration": next_iter, "hyperparameters": algo_info.get("hyperparameters", {})})
    events = _emit({"events": events}, "code_generated",
                   f"step3_ml_iter{next_iter}.py written — [{chosen_strategy}] {chosen_algo}",
                   data=result["code"])

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
    """Execute the current optimization iteration script."""
    _set_session(state["session_id"])
    script_name = state.get("opt_script_name", "")
    iteration = state.get("optimization_iteration", 1)

    if not script_name:
        err = f"opt_script_name is empty for iteration {iteration} — optimizer node may not have run."
        logger.error(err)
        events = _emit(state, "execution_error", err)
        return {**state, "opt_output": "", "opt_error": err, "status": "failed", "events": events}

    events = _emit(state, "executing", f"Running {script_name} (iteration {iteration})...")

    script_path = GENERATED_CODE_DIR / script_name
    _session_log = OUTPUTS_DIR / (state.get("session_id") or "default") / "log.json"
    result = run_script_with_retry(
        script_name,
        fix_callback=optimizer.make_fix_callback(script_path),
        max_retries=MAX_RETRIES,
        log_path=_session_log,
    )

    if result.success:
        # Read the final file content — fix_callback may have rewritten it, so disk is authoritative.
        working_code = (GENERATED_CODE_DIR / script_name).read_text() if script_name else ""
        events = _emit({"events": events}, "execution_success",
                       f"Iteration {iteration} succeeded.", data=result.stdout[:3000])
        return {**state, "opt_output": result.stdout, "opt_error": "",
                "last_opt_working_code": working_code, "events": events}

    events = _emit({"events": events}, "execution_error",
                   f"Iteration {iteration} failed after {MAX_RETRIES} retries.", data=result.stderr)
    return {**state, "opt_output": "", "opt_error": result.stderr, "status": "failed", "events": events}


def node_finalize_best(state: PipelineState) -> PipelineState:
    """Copy best model to model.pkl and save optimization_history.json."""
    best = state.get("best_model", {})
    history = state.get("optimization_history", [])

    events = _emit(state, "agent_thinking", "Finalizing best model across all iterations...")

    session_out = OUTPUTS_DIR / (state.get("session_id") or "default")
    session_out.mkdir(parents=True, exist_ok=True)

    best_path  = best.get("model_path", "")
    final_path = str(session_out / "model.pkl")

    if best_path and Path(best_path).exists():
        shutil.copy2(best_path, final_path)
        logger.info("Copied best model %s → %s", best_path, final_path)
    else:
        logger.warning("Best model path not found: %s", best_path)

    history_path = session_out / "optimization_history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str))

    summary_lines = [
        f"  iter {h['iteration']}: {h['algorithm']} — {h['primary_metric']}={h['primary_score']:.4f}"
        for h in history
    ]
    summary = "\n".join(summary_lines)

    events = _emit(
        {"events": events}, "optimization_complete",
        f"Optimization done. Best: {best.get('algorithm')} "
        f"({best.get('primary_metric')}={best.get('primary_score', 0):.4f})\n{summary}",
        data={"best_model": best, "history": history},
    )
    return {**state, "events": events}


def node_evaluation(state: PipelineState) -> PipelineState:
    _set_session(state["session_id"])
    events = _emit(state, "agent_thinking", "Evaluation Agent is analyzing best model performance...")

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
    # ── Collect token usage BEFORE completed so the frontend stream is still open ──
    session_id = state.get("session_id", "default")
    token_usage = _get_token_usage(session_id)
    if token_usage:
        events = _emit({"events": events}, "token_usage", "Token usage summary ready.", data=token_usage)

    # ── Auto-validation on held-out 5% ───────────────────────────────
    val_csv = state.get("validation_csv_path", "")
    if val_csv and Path(val_csv).exists():
        events = _emit({"events": events}, "agent_thinking",
                       "Running validation agent on held-out 5% data...")
        try:
            from agents import testing_agent as _ta
            model_path = str(OUTPUTS_DIR / session_id / "model.pkl")
            training_code = state.get("last_opt_working_code", "")
            val_result = _ta.run(
                test_csv_path=val_csv,
                model_path=model_path,
                task_type=state["task_type"],
                target_column=state.get("target_column"),
                training_code=training_code,
                session_id=session_id + "_val",
            )
            if "error" not in val_result:
                events = _emit({"events": events}, "validation_results",
                               "Held-out validation complete.", data=val_result)
                logger.info("Validation complete for session %s", session_id)
            else:
                _err_msg = val_result.get("error", "unknown")
                val_result = {}
                events = _emit({"events": events}, "validation_error",
                               f"Validation failed: {_err_msg}")
        except Exception as exc:
            val_result = {}
            logger.error("Validation agent error: %s", exc, exc_info=True)
            events = _emit({"events": events}, "validation_error",
                           f"Validation agent error: {exc}")
    else:
        val_result = {}

    # ── completed must be last — frontend closes the SSE stream on this event ──
    events = _emit({"events": events}, "completed", "Pipeline finished successfully!")

    return {**state, "evaluation": evaluation, "token_usage": token_usage,
            "validation_results": val_result, "status": "completed", "events": events}


# ─────────────────────────── Edges ────────────────────────────

def edge_after_understanding_exec(state: PipelineState) -> Literal["task_profiling", "__end__"]:
    return "__end__" if state.get("understanding_error") else "task_profiling"


def edge_after_profiling_exec(state: PipelineState) -> Literal["data_analysis", "__end__"]:
    return "__end__" if state.get("profiling_error") else "data_analysis"


def edge_after_analysis_exec(state: PipelineState) -> Literal["human_approval", "__end__"]:
    return "__end__" if state.get("analysis_error") else "human_approval"


def edge_after_human_approval(state: PipelineState) -> Literal["ml_engineering", "human_approval"]:
    return "ml_engineering" if state.get("human_approved") else "human_approval"


def edge_after_ml_exec(state: PipelineState) -> Literal["track_metrics", "__end__"]:
    return "__end__" if state.get("ml_error") else "track_metrics"


def edge_optimization_loop(state: PipelineState) -> Literal["optimizer", "finalize_best"]:
    # Stop only when the last model has been fully tuned AND no more models to try.
    tune_done   = state.get("current_algo_tune_count", 0) >= MAX_TUNE_ITERATIONS
    models_done = state.get("models_tried", 0) >= MAX_OPTIMIZATION_LOOPS
    if tune_done and models_done:
        return "finalize_best"
    return "optimizer"


def edge_after_opt_exec(state: PipelineState) -> Literal["track_metrics", "__end__"]:
    return "__end__" if state.get("opt_error") else "track_metrics"


def edge_after_track(state: PipelineState) -> Literal["optimization_loop", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "optimization_loop"


# ─────────────────────────── Graph ────────────────────────────

def _build_workflow():
    workflow = StateGraph(PipelineState)

    workflow.add_node("data_understanding",     node_data_understanding)
    workflow.add_node("execute_understanding",  node_execute_understanding)
    workflow.add_node("task_profiling",         node_task_profiling)
    workflow.add_node("execute_profiling",      node_execute_profiling)
    workflow.add_node("data_analysis",          node_data_analysis)
    workflow.add_node("execute_analysis",       node_execute_analysis)
    workflow.add_node("human_approval",         node_human_approval)
    workflow.add_node("ml_engineering",         node_ml_engineering)
    workflow.add_node("execute_ml",             node_execute_ml)
    workflow.add_node("track_metrics",          node_track_metrics)
    workflow.add_node("optimization_loop",      lambda s: s)
    workflow.add_node("optimizer",              node_optimizer)
    workflow.add_node("execute_optimization",   node_execute_optimization)
    workflow.add_node("finalize_best",          node_finalize_best)
    workflow.add_node("evaluation",             node_evaluation)

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
    workflow.add_conditional_edges(
        "human_approval",
        edge_after_human_approval,
        {"ml_engineering": "ml_engineering", "human_approval": "human_approval"},
    )
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
        {"track_metrics": "track_metrics", "__end__": END},
    )
    workflow.add_edge("finalize_best", "evaluation")
    workflow.add_edge("evaluation", END)

    return workflow.compile(
        checkpointer=_checkpointer,
        interrupt_before=["human_approval"],
    )


def build_graph():
    return _build_workflow()


graph = build_graph()


def create_initial_state(
    dataset_path: str, task_type: str, target_column: str = None, session_id: str = "",
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
    )
