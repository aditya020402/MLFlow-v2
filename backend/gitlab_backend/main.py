"""
GitLab Backend — standalone FastAPI service (port 8001).

Responsibilities:
  1. Register GitLab projects to monitor.
  2. Poll those projects every N seconds for issues labelled ml-ready.
  3. For each open issue:
       a. Download the CSV attachment.
       b. Parse the issue body as instructions.
       c. Run the ML agent pipeline (same graph as the main backend).
       d. Post progress comments back to the issue.
       e. Upload model.pkl + metrics as attachments and post a final summary.
       f. Close the issue and label it ml-complete (or ml-failed on error).

User-facing issue format:
  ## Task Type
  classification

  ## Target Variable
  `gender`

  ## Business Context
  ...

  ## ML Requirements
  - prefer recall over precision
  - try XGBoost

  ## Data Notes
  - column X has erroneous zeros

  [data.csv](/uploads/hash/data.csv)   ← drag-and-drop attachment
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from gitlab_backend import db, instructions as instr_parser, pipeline
from gitlab_backend.client import GitLabClient, LABEL_READY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GITLAB] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GITLAB_BASE_URL       = os.environ["GITLAB_BASE_URL"]
GITLAB_DEFAULT_TOKEN  = os.environ["GITLAB_ACCESS_TOKEN"]
POLLING_INTERVAL      = int(os.environ.get("GITLAB_POLLING_INTERVAL", "60"))

UPLOADS_DIR = Path(__file__).parent.parent / "uploads" / "gitlab"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# One executor — pipeline runs sequentially to avoid shared-file conflicts
_executor = ThreadPoolExecutor(max_workers=1)
_active_sessions: set[str] = set()   # session_ids currently running


# ── Lifespan (start/stop poller) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app):
    db.init_db()
    logger.info("GitLab backend started. Polling every %ds.", POLLING_INTERVAL)
    task = asyncio.create_task(_poll_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    _executor.shutdown(wait=False)


app = FastAPI(
    title="GitLab ML Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    project_url:           str
    access_token:          str = GITLAB_DEFAULT_TOKEN
    branch:                str = "main"
    polling_interval_sec:  int = POLLING_INTERVAL
    csv_filename:          str = "data.csv"
    instructions_filename: str = "instructions.md"
    owner_label:           str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "polling_interval": POLLING_INTERVAL}


@app.post("/gitlab/register")
def register_project(body: RegisterRequest):
    """Register a GitLab project for the poller to monitor."""
    # Derive namespace/path from the URL, then look up the project ID
    try:
        path = _namespace_from_url(body.project_url)
        client = GitLabClient(GITLAB_BASE_URL, body.access_token, project_id=0)
        pid = client.lookup_project_id(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not reach GitLab project: {exc}")

    db.upsert_project(
        project_url=body.project_url,
        gitlab_project_id=pid,
        access_token=body.access_token,
        branch=body.branch,
        polling_interval_sec=body.polling_interval_sec,
        csv_filename=body.csv_filename,
        instructions_filename=body.instructions_filename,
        owner_label=body.owner_label,
    )
    logger.info("Registered project %s (id=%d)", body.project_url, pid)
    return {"status": "registered", "gitlab_project_id": pid, "project_url": body.project_url}


@app.get("/gitlab/projects")
def list_projects():
    return {"projects": db.get_all_projects()}


@app.get("/gitlab/runs")
def list_runs():
    return {"runs": db.get_all_runs()}


@app.get("/gitlab/runs/{session_id}")
def get_run(session_id: str):
    run = db.get_run(session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


# ── Polling loop ──────────────────────────────────────────────────────────────

async def _poll_loop():
    while True:
        await asyncio.sleep(POLLING_INTERVAL)
        try:
            await _poll_once()
        except Exception as exc:
            logger.error("Poll cycle error: %s", exc, exc_info=True)


async def _poll_once():
    projects = db.get_all_projects()
    if not projects:
        return

    loop = asyncio.get_running_loop()
    for project in projects:
        try:
            client = GitLabClient(
                GITLAB_BASE_URL,
                project["access_token"],
                project["gitlab_project_id"],
                project_path=_namespace_from_url(project["project_url"]),
            )
            issues = client.get_open_issues(label=LABEL_READY)
            logger.info(
                "Project %s — found %d %s issue(s)",
                project["project_url"], len(issues), LABEL_READY,
            )
            for issue in issues:
                await _dispatch_issue(loop, project, client, issue)
        except Exception as exc:
            logger.error("Error polling project %s: %s", project["project_url"], exc)


async def _dispatch_issue(loop, project: dict, client: GitLabClient, issue: dict):
    """Validate the issue, claim it, and submit the pipeline to the executor."""
    issue_iid = issue["iid"]
    body      = issue.get("description") or ""
    title     = issue.get("title", "")
    author    = (issue.get("author") or {}).get("username", "unknown")

    # ── Parse instructions ──────────────────────────────────────────────
    parsed  = instr_parser.parse(body)
    err_msg = instr_parser.validation_error(parsed)
    if err_msg:
        logger.warning("Issue #%d validation failed: %s", issue_iid, err_msg)
        client.post_comment(
            issue_iid,
            f"❌ **Could not start pipeline.**\n\n{err_msg}",
        )
        client.update_issue(issue_iid, remove_labels=LABEL_READY)
        return

    # ── Find CSV attachment ─────────────────────────────────────────────
    csv_match = GitLabClient.extract_attachment(body, "csv")
    if not csv_match:
        client.post_comment(
            issue_iid,
            "❌ **No CSV attachment found.**\n\n"
            "Please attach a `.csv` file to the issue description by dragging it in.",
        )
        client.update_issue(issue_iid, remove_labels=LABEL_READY)
        return

    csv_filename, csv_url = csv_match

    # ── Claim the issue (UNIQUE guard against double-processing) ────────
    session_id = f"gitlab-{uuid.uuid4()}"
    claimed = db.try_claim_issue(
        session_id         = session_id,
        gitlab_project_id  = project["gitlab_project_id"],
        issue_iid          = issue_iid,
        issue_title        = title,
        triggered_by       = author,
        csv_filename       = csv_filename,
        task_type          = parsed["task_type"],
        target_column      = parsed.get("target_column") or "",
    )
    if not claimed:
        logger.info("Issue #%d already claimed — skipping.", issue_iid)
        return

    # ── Flip label: ml-ready → ml-processing ───────────────────────────
    client.claim_issue(issue_iid)
    client.post_comment(
        issue_iid,
        _start_comment(
            csv_filename,
            parsed["task_type"],
            parsed.get("target_column"),
            author,
        ),
    )

    logger.info(
        "Dispatching issue #%d → session %s  task=%s target=%s",
        issue_iid, session_id, parsed["task_type"], parsed.get("target_column"),
    )

    # ── Submit to thread pool ───────────────────────────────────────────
    _active_sessions.add(session_id)
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        _executor,
        lambda: _run_issue_pipeline(
            project, client, issue_iid, session_id,
            csv_url, csv_filename, parsed, loop,
        ),
    )

    def _on_done(fut):
        _active_sessions.discard(session_id)
        exc = fut.exception()
        if exc:
            logger.error("Session %s raised: %s", session_id, exc, exc_info=exc)

    future.add_done_callback(_on_done)


# ── Pipeline runner (executed in thread pool) ─────────────────────────────────

def _run_issue_pipeline(
    project: dict,
    client: GitLabClient,
    issue_iid: int,
    session_id: str,
    csv_url: str,
    csv_filename: str,
    parsed: dict,
    loop: asyncio.AbstractEventLoop,
):
    def post(msg: str):
        """Thread-safe GitLab comment poster."""
        try:
            client.post_comment(issue_iid, msg)
        except Exception as exc:
            logger.warning("Failed to post comment: %s", exc)

    try:
        # ── Download CSV ────────────────────────────────────────────────
        session_uploads = UPLOADS_DIR / session_id
        session_uploads.mkdir(parents=True, exist_ok=True)
        local_csv = session_uploads / csv_filename
        client.download_attachment(csv_url, local_csv)
        logger.info("[%s] CSV saved: %s (%d bytes)", session_id, local_csv, local_csv.stat().st_size)

        # ── Run pipeline ────────────────────────────────────────────────
        final_state = pipeline.run(
            dataset_path   = str(local_csv),
            task_type      = parsed["task_type"],
            target_column  = parsed.get("target_column"),
            human_feedback = parsed["human_feedback"],
            session_id     = session_id,
            progress_cb    = post,
        )

        # ── Upload results ──────────────────────────────────────────────
        _upload_and_summarise(client, issue_iid, session_id, final_state)

        db.update_run_status(session_id, "completed")
        client.complete_issue(issue_iid)
        logger.info("[%s] Issue #%d completed successfully.", session_id, issue_iid)

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", session_id, exc, exc_info=True)
        db.update_run_status(session_id, "failed")
        try:
            client.post_comment(
                issue_iid,
                f"❌ **Pipeline failed.**\n\n```\n{exc}\n```\n\n"
                "Check the backend logs for details.",
            )
            client.fail_issue(issue_iid)
        except Exception:
            pass


# ── Results upload & final comment ────────────────────────────────────────────

def _upload_and_summarise(
    client: GitLabClient,
    issue_iid: int,
    session_id: str,
    final_state: dict,
):
    model_link      = ""
    history_link    = ""

    # Upload model.pkl
    model_path = OUTPUTS_DIR / "model.pkl"
    if model_path.exists():
        try:
            up = client.upload_file(model_path)
            model_link = f"\n\n[⬇ Download model.pkl]({up['url']})"
        except Exception as exc:
            logger.warning("model.pkl upload failed: %s", exc)

    # Upload optimization_history.json
    hist_path = OUTPUTS_DIR / "optimization_history.json"
    if hist_path.exists():
        try:
            up = client.upload_file(hist_path)
            history_link = f"  [optimization history]({up['url']})"
        except Exception as exc:
            logger.warning("history upload failed: %s", exc)

    evaluation = final_state.get("evaluation", {})
    best       = final_state.get("best_model", {})
    history    = final_state.get("optimization_history", [])

    comment = _results_comment(evaluation, best, history, model_link, history_link, session_id)
    client.post_comment(issue_iid, comment)


def _results_comment(
    evaluation: dict,
    best: dict,
    history: list,
    model_link: str,
    history_link: str,
    session_id: str,
) -> str:
    verdict = evaluation.get("verdict", "unknown").upper()
    verdict_icon = "✅" if verdict == "PASS" else "⚠️"

    algo    = best.get("algorithm", "unknown")
    metrics = best.get("metrics", {})
    primary = best.get("primary_metric", "score")
    _skip   = {
        "algorithm", "iteration", "task_type", "model_path",
        "train_samples", "test_samples", "hyperparameters", "strategy",
    }

    # Metrics table
    metric_rows = [f"| {primary.upper()} | **{best.get('primary_score', 0):.4f}** |"]
    for k, v in metrics.items():
        if k in _skip or k == primary or not isinstance(v, float):
            continue
        metric_rows.append(f"| {k.upper()} | {v:.4f} |")

    metric_table = (
        "| Metric | Value |\n|--------|-------|\n" + "\n".join(metric_rows)
        if metric_rows else "_No metrics available._"
    )

    # Optimization history table
    hist_rows = []
    for h in history:
        strat = h.get("strategy", "")
        itr   = h.get("iteration", "?")
        halgo = h.get("algorithm", "?")
        sc    = h.get("primary_score", 0)
        hist_rows.append(f"| {itr} | {strat} | `{halgo}` | {sc:.4f} |")
    history_table = (
        "| Iter | Strategy | Algorithm | Score |\n"
        "|------|----------|-----------|-------|\n"
        + "\n".join(hist_rows)
    ) if hist_rows else "_No history._"

    summary   = evaluation.get("summary", "")
    strengths = "\n".join(f"- {s}" for s in evaluation.get("strengths", []))
    weaknesses= "\n".join(f"- {s}" for s in evaluation.get("weaknesses", []))
    suggestions="\n".join(f"- {s}" for s in evaluation.get("suggestions", []))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""## {verdict_icon} Pipeline Complete — {verdict}

**Best Model:** `{algo}`
**Session:** `{session_id}`
**Completed:** {ts}

### Performance Metrics

{metric_table}

### Optimization History

{history_table}

---

**Summary:** {summary}

{"### Strengths" + chr(10) + strengths if strengths else ""}

{"### Weaknesses" + chr(10) + weaknesses if weaknesses else ""}

{"### Suggestions" + chr(10) + suggestions if suggestions else ""}

---
{model_link}  {history_link}

*Generated by ML Pipeline Agent*"""


# ── Formatting helpers ────────────────────────────────────────────────────────

def _start_comment(csv_filename: str, task_type: str, target_column: str | None, author: str) -> str:
    target_line = f"**Target column:** `{target_column}`  \n" if target_column else ""
    return (
        f"🚀 **Pipeline started.**\n\n"
        f"**Dataset:** `{csv_filename}`  \n"
        f"**Task:** `{task_type}`  \n"
        f"{target_line}"
        f"**Triggered by:** @{author}  \n\n"
        f"Progress updates will be posted here as the pipeline runs."
    )


def _namespace_from_url(url: str) -> str:
    """Extract 'namespace/project' from a GitLab project URL."""
    url = url.rstrip("/")
    # https://gitlab.com/group/project  →  group/project
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse namespace from URL: {url}")
    return "/".join(parts[-2:])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GITLAB_PORT", "8001"))
    uvicorn.run("gitlab_backend.main:app", host="0.0.0.0", port=port, reload=True)
