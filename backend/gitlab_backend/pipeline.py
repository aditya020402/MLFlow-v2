"""Runs the shared ML agent graph for a GitLab-sourced issue.

Wraps the existing LangGraph pipeline:
  - Phase 1: data understanding → analysis  (pauses at human_approval)
  - Auto-approve using the parsed instructions as human_feedback
  - Phase 2: ML engineering → optimization → evaluation
  - Fires progress_callback at key milestones so the caller can post GitLab comments
"""

import csv
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.agent_graph import build_graph, create_initial_state

logger = logging.getLogger(__name__)

# Events we surface as GitLab progress comments
_MILESTONE_EVENTS = {
    "execution_success",
    "analysis_data",
    "algorithm_selected",
    "best_model_updated",
    "optimization_complete",
    "evaluation_done",
    "execution_error",
}


def _split_95_5(source: Path, train: Path, val: Path, seed: int = 42) -> tuple[int, int]:
    with open(source, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    n_val = max(1, int(len(rows) * 0.05))
    rng = random.Random(seed)
    val_indices = set(rng.sample(range(len(rows)), n_val))
    with open(train, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(rows):
            if i not in val_indices:
                w.writerow(row)
    with open(val, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(rows):
            if i in val_indices:
                w.writerow(row)
    return len(rows) - n_val, n_val


def run(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    human_feedback: str,
    session_id: str,
    progress_cb,   # callable(str) — posts a comment to the GitLab issue
    validate: bool = False,
) -> dict:
    """
    Execute the full ML pipeline and return the final LangGraph state.

    progress_cb is called at key milestones with a markdown-formatted string.
    Runs synchronously — call this from a thread pool.
    """
    # ── Optional 95/5 split ───────────────────────────────────────────────
    validation_csv_path = ""
    if validate:
        src = Path(dataset_path)
        val_path = src.parent / f"{src.stem}_validation.csv"
        try:
            n_train, n_val = _split_95_5(src, src, val_path)
            validation_csv_path = str(val_path)
            logger.info("[%s] 95/5 split: %d train / %d val rows", session_id, n_train, n_val)
            progress_cb(f"📊 **Validation split:** {n_train} training / {n_val} held-out rows.")
        except Exception as exc:
            logger.error("[%s] CSV split failed: %s", session_id, exc)
            progress_cb("⚠️ Validation split failed — continuing without validation.")

    graph   = build_graph()
    state   = create_initial_state(dataset_path, task_type, target_column,
                                   session_id=session_id,
                                   validation_csv_path=validation_csv_path)
    config  = {"configurable": {"thread_id": session_id}}

    seen      = 0
    last      = {}
    posted    : set[str] = set()

    # ── Phase 1: understanding → analysis (halts at human_approval) ──────
    logger.info("[%s] Phase 1 starting…", session_id)
    for chunk in graph.stream(state, config=config, stream_mode="values"):
        last = chunk
        seen = _process_events(last.get("events", []), seen, posted, progress_cb, session_id)

    # ── Auto-approve ──────────────────────────────────────────────────────
    if last.get("status") == "awaiting_approval":
        progress_cb(
            "📋 **Analysis complete.** Auto-approving and starting ML training…\n\n"
            f"> Instructions injected as agent feedback."
        )
        graph.update_state(
            config,
            {
                "human_approved":  True,
                "status":          "running",
                "human_feedback":  human_feedback,
            },
            as_node="human_approval",
        )

        # ── Phase 2: ML → optimization → evaluation ───────────────────────
        logger.info("[%s] Phase 2 starting…", session_id)
        for chunk in graph.stream(None, config=config, stream_mode="values"):
            last = chunk
            seen = _process_events(last.get("events", []), seen, posted, progress_cb, session_id)

    logger.info("[%s] Pipeline finished — status=%s", session_id, last.get("status"))
    return last


# ── Internal helpers ──────────────────────────────────────────────────────────

def _process_events(
    events: list,
    seen: int,
    posted: set,
    progress_cb,
    session_id: str,
) -> int:
    for evt in events[seen:]:
        evt_type = evt.get("type", "")
        message  = evt.get("message", "")
        data     = evt.get("data") or {}

        if evt_type not in _MILESTONE_EVENTS:
            continue

        # Deduplicate by (type, message) — same event can appear in multiple chunks
        key = f"{evt_type}:{message[:80]}"
        if key in posted:
            continue
        posted.add(key)

        comment = _format_event(evt_type, message, data)
        if comment:
            try:
                progress_cb(comment)
            except Exception as exc:
                logger.warning("[%s] progress_cb failed: %s", session_id, exc)

    return len(events)


def _format_event(evt_type: str, message: str, data: dict) -> str | None:
    if evt_type == "execution_success":
        msg_low = message.lower()
        if "general understanding" in msg_low:
            return "🔍 **General data understanding complete.**"
        if "task profiling" in msg_low:
            return "📈 **Task-specific profiling complete.**"
        return None  # other execution_success events (step2, step3) not worth a comment

    if evt_type == "analysis_data":
        plot_insights = (data or {}).get("plot_insights", [])
        ml_recs = (data or {}).get("ml_model_recommendations", {})
        lines = ["📊 **Data analysis complete.**"]
        if plot_insights:
            lines.append("\n**Key insights from the data:**")
            lines.extend(f"- {i}" for i in plot_insights[:4])
        if ml_recs:
            hints = ml_recs.get("algorithm_hints", [])
            challenges = ml_recs.get("expected_challenges", [])
            if hints:
                lines.append("\n**Algorithm recommendations:**")
                lines.extend(f"- {h}" for h in hints[:3])
            if challenges:
                lines.append("\n**Expected challenges:**")
                lines.extend(f"- {c}" for c in challenges[:3])
        lines.append("\nStarting ML training…")
        return "\n".join(lines)

    if evt_type == "algorithm_selected":
        algo   = (data or {}).get("algorithm", "?")
        reason = (data or {}).get("reason", "")
        short  = reason[:200] + "…" if len(reason) > 200 else reason
        return f"🤖 **Algorithm selected:** `{algo}`\n> {short}"

    if evt_type == "best_model_updated":
        algo    = data.get("algorithm", "?")
        score   = data.get("primary_score", 0)
        met     = data.get("primary_metric", "score")
        itr     = data.get("iteration", "?")
        metrics = data.get("metrics", {})
        extra   = _fmt_secondary(metrics, met)
        return (
            f"🏆 **New best model (iter {itr}):** `{algo}`\n"
            f"> {met} = **{score:.4f}**{extra}"
        )

    if evt_type == "optimization_complete":
        return f"⚙️ **Optimization complete.** {message}"

    if evt_type == "evaluation_done":
        best    = (data or {}).get("best_model", {})
        algo    = best.get("algorithm", "?")
        score   = best.get("primary_score", 0)
        met     = best.get("primary_metric", "score")
        verdict = (data or {}).get("verdict", "?").upper()
        summary = (data or {}).get("summary", "")
        icon    = "✅" if verdict == "PASS" else "⚠️"
        return (
            f"{icon} **Evaluation complete — {verdict}**\n\n"
            f"**Best model:** `{algo}` — {met} = **{score:.4f}**\n"
            f"> {summary}"
        )

    if evt_type == "execution_error":
        return f"⚠️ **Execution error:** {message}"

    return None


def _fmt_secondary(metrics: dict, primary_key: str) -> str:
    _skip = {
        "algorithm", "iteration", "task_type", "model_path",
        "train_samples", "test_samples", "hyperparameters",
        "strategy", primary_key,
    }
    parts = [
        f"{k}={v:.4f}" for k, v in metrics.items()
        if k not in _skip and isinstance(v, float)
    ]
    return ("  |  " + "  |  ".join(parts[:3])) if parts else ""
