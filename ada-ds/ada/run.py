"""Core orchestrator for the ada-ds pip package.

Usage (Python API):
    from ada import run_pipeline
    result = run_pipeline(
        dataset="path/to/data.csv",
        instructions="path/to/instructions.md",
        output_dir="./my_output",   # optional
    )
"""

import csv
import json
import logging
import os
import random
import shutil
import string
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _make_session_id(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _split_csv_95_5(
    source_path: Path,
    train_path: Path,
    val_path: Path,
    seed: int = 42,
) -> tuple[int, int]:
    """Split a CSV into 95 % train / 5 % validation rows. Returns (n_train, n_val)."""
    with open(source_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    rng = random.Random(seed)
    rng.shuffle(rows)
    split = max(1, int(len(rows) * 0.95))
    train_rows = rows[:split]
    val_rows   = rows[split:]

    def _write(path: Path, data: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    _write(train_path, train_rows)
    _write(val_path,   val_rows)
    return len(train_rows), len(val_rows)


def run_pipeline(
    dataset: str,
    instructions: str,
    output_dir: str = "./ada_output",
    env_file: str = ".env",
    verbose: bool = True,
) -> dict:
    """
    Run the full Ada ML pipeline.

    Parameters
    ----------
    dataset      : path to the input CSV file
    instructions : path to the .md instructions file
    output_dir   : root directory; each run creates a session subfolder inside it
    env_file     : path to .env file with OPENAI_API_KEY etc.
    verbose      : if True, show clean progress events (no internal log noise)

    Returns
    -------
    dict with keys: status, model_path, evaluation, validation_results,
                    summary_path, token_usage, session_id
    """
    # ── Silence LangChain / LangGraph deprecation warnings ───────
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
    warnings.filterwarnings("ignore", category=UserWarning, module="langgraph")

    # ── Logging: only show clean print() events, suppress internal noise ──
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    # Runner retry messages and agent INFO logs are already represented
    # by the [OK]/[ERR]/[>>>] print lines — suppress them to avoid duplication.
    logging.getLogger("ada.tools.runner").setLevel(logging.ERROR)
    logging.getLogger("ada.graph.agent_graph").setLevel(logging.ERROR)
    logging.getLogger("ada.agents").setLevel(logging.ERROR)

    # ── Environment ───────────────────────────────────────────────
    if Path(env_file).exists():
        load_dotenv(env_file)
    elif Path(".env").exists():
        load_dotenv(".env")

    # ── Validate inputs ───────────────────────────────────────────
    dataset_path = Path(dataset).resolve()
    instructions_path = Path(instructions).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")

    # ── Parse instructions ────────────────────────────────────────
    from ada.instructions import parse, validation_error
    parsed = parse(instructions_path.read_text(encoding="utf-8"))
    err = validation_error(parsed)
    if err:
        raise ValueError(f"Instructions error: {err}")

    task_type      = parsed["task_type"]
    target_column  = parsed.get("target_column")
    validate       = parsed.get("validate", False)
    human_feedback = parsed.get("human_feedback", "")

    # ── Session folder — everything for this run goes here ────────
    import ada.paths as _paths_mod
    out_root   = Path(output_dir).resolve()
    session_id = _make_session_id()
    session_dir = out_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Point all path globals at the session subfolder
    _paths_mod.OUTPUTS_DIR        = out_root          # node_finalize_best appends session_id itself
    _paths_mod.GENERATED_CODE_DIR = session_dir / "generated_code"
    _paths_mod.UPLOADS_DIR        = session_dir / "data"

    _paths_mod.GENERATED_CODE_DIR.mkdir(parents=True, exist_ok=True)
    _paths_mod.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Copy dataset into session data dir ────────────────────────
    managed_dataset = _paths_mod.UPLOADS_DIR / dataset_path.name
    shutil.copy2(dataset_path, managed_dataset)
    working_dataset = str(managed_dataset)

    # ── Optional 95/5 validation split ───────────────────────────
    validation_csv_path = ""
    if validate:
        val_path = _paths_mod.UPLOADS_DIR / "validation.csv"
        try:
            n_train, n_val = _split_csv_95_5(managed_dataset, managed_dataset, val_path)
            validation_csv_path = str(val_path)
            print(f"[ADA] Validation split: {n_train} train / {n_val} validation rows")
        except Exception as exc:
            logger.warning("CSV split failed (continuing without validation): %s", exc)

    # ── Build and run graph ───────────────────────────────────────
    from ada.graph.agent_graph import build_graph, create_initial_state

    graph = build_graph()
    initial_state = create_initial_state(
        dataset_path=working_dataset,
        task_type=task_type,
        target_column=target_column,
        session_id=session_id,
        validation_csv_path=validation_csv_path,
    )
    if human_feedback:
        initial_state = {**initial_state, "human_feedback": human_feedback}

    print(f"\n[ADA] Session : {session_id}")
    print(f"[ADA] Task    : {task_type}  |  Target: {target_column or 'none (unsupervised)'}")
    print(f"[ADA] Output  : {session_dir}\n")

    t0 = time.time()
    config = {"configurable": {"thread_id": session_id}}
    final_state = None
    for chunk in graph.stream(initial_state, config=config):
        for _, state_update in chunk.items():
            final_state = state_update

    elapsed = time.time() - t0

    if final_state is None:
        raise RuntimeError("Graph produced no output.")

    # ── Locate outputs — all already inside session_dir ──────────
    # node_finalize_best writes best_{Algo}_{metric}_{score}.pkl into session_dir
    best_model = final_state.get("best_model", {})
    model_filename = final_state.get("final_model_filename", "") or "model.pkl"
    model_path = session_dir / model_filename
    if not model_path.exists():
        # fallback: pick any best_*.pkl in session_dir
        candidates = list(session_dir.glob("best_*.pkl"))
        if candidates:
            model_path = candidates[0]

    # ── Write summary.json into session_dir ───────────────────────
    evaluation  = final_state.get("evaluation", {})
    val_results = final_state.get("validation_results", {})
    token_usage = final_state.get("token_usage", {})

    summary = {
        "session_id":           session_id,
        "task_type":            task_type,
        "target_column":        target_column,
        "dataset":              str(dataset_path),
        "status":               final_state.get("status", "unknown"),
        "elapsed_seconds":      round(elapsed, 1),
        "best_model": {
            "algorithm":        best_model.get("algorithm"),
            "primary_metric":   best_model.get("primary_metric"),
            "score":            best_model.get("primary_score"),
            "metrics":          best_model.get("metrics", {}),
        },
        "evaluation":           evaluation,
        "validation":           val_results.get("metrics", {}) if val_results else {},
        "optimization_history": final_state.get("optimization_history", []),
        "token_usage":          token_usage,
        "output_dir":           str(session_dir),
        "model_path":           str(model_path) if model_path.exists() else None,
    }

    summary_path = session_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ── Print final summary ───────────────────────────────────────
    algo = best_model.get("algorithm", "?")
    pm   = best_model.get("primary_metric", "?")
    sc   = best_model.get("primary_score", 0) or 0

    print("\n" + "=" * 60)
    print("ADA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model   : {algo}")
    print(f"  {pm:14s} : {sc:.4f}")
    if val_results and val_results.get("metrics"):
        vm = val_results["metrics"]
        val_line = "  |  ".join(f"{k}={v:.4f}" for k, v in vm.items() if isinstance(v, float))
        print(f"  Validation   : {val_line}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print(f"  Model        : {model_path.name if model_path.exists() else 'NOT FOUND'}")
    print(f"  Model path   : {model_path if model_path.exists() else ''}")
    print(f"  Summary      : {summary_path}")
    print(f"  Session dir  : {session_dir}")
    print("=" * 60 + "\n")

    return {
        "status":             final_state.get("status"),
        "model_path":         str(model_path) if model_path.exists() else None,
        "evaluation":         evaluation,
        "validation_results": val_results,
        "summary_path":       str(summary_path),
        "token_usage":        token_usage,
        "session_id":         session_id,
        "session_dir":        str(session_dir),
    }
