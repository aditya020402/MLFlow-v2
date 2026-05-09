"""Runs the shared ML agent graph for a GitLab-sourced issue.

Wraps the existing LangGraph pipeline:
  - Phase 1: data understanding → analysis  (pauses at human_approval)
  - Auto-approve using the parsed instructions as human_feedback
  - Phase 2: ML engineering → optimization → evaluation
  - Fires progress_callback at key milestones so the caller can post GitLab comments
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.agent_graph import build_graph, create_initial_state

logger = logging.getLogger(__name__)

# Events we surface as GitLab progress comments
_MILESTONE_EVENTS = {
    "execution_success",
    "analysis_data",
    "best_model_updated",
    "optimization_complete",
    "evaluation_done",
    "execution_error",
}


def run(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    human_feedback: str,
    session_id: str,
    progress_cb,   # callable(str) — posts a comment to the GitLab issue
) -> dict:
    """
    Execute the full ML pipeline and return the final LangGraph state.

    progress_cb is called at key milestones with a markdown-formatted string.
    Runs synchronously — call this from a thread pool.
    """
    graph   = build_graph()
    state   = create_initial_state(dataset_path, task_type, target_column)
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
    if evt_type == "execution_success" and "step1" in message.lower():
        return "🔍 **Data understanding complete.**"

    if evt_type == "analysis_data":
        return "📊 **Data analysis complete.** Starting ML training…"

    if evt_type == "best_model_updated":
        algo  = data.get("algorithm", "?")
        score = data.get("primary_score", 0)
        met   = data.get("primary_metric", "score")
        itr   = data.get("iteration", "?")
        metrics = data.get("metrics", {})
        extra = _fmt_secondary(metrics, met)
        return (
            f"🏆 **New best model (iter {itr}):** `{algo}`\n"
            f"> {met} = **{score:.4f}**{extra}"
        )

    if evt_type == "optimization_complete":
        return f"⚙️ **Optimization complete.** {message}"

    if evt_type == "execution_error":
        return f"⚠️ Execution error: {message}"

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
