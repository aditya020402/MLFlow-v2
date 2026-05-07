"""
FastAPI backend for the Agentic AI Data Science System.

Endpoints:
  POST /upload              — upload dataset, receive session_id
  POST /start               — start the pipeline
  GET  /stream/{id}         — SSE stream of agent events
  GET  /state/{id}          — current pipeline state snapshot
  POST /approve/{id}        — inject human approval to continue past analysis
  GET  /code/{id}/{step}    — retrieve generated code for a step (understanding|analysis|ml)
  GET  /download/{id}/model — download best model.pkl
  GET  /health              — health check
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import csv
import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── path setup ──
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from graph.agent_graph import create_initial_state, graph as _shared_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
logger = logging.getLogger(__name__)

UPLOADS_DIR = BASE_DIR / "uploads"
GENERATED_CODE_DIR = BASE_DIR / "generated_code"
OUTPUTS_DIR = BASE_DIR / "outputs"
SESSIONS_DIR = BASE_DIR / "outputs" / "sessions"

for _d in [UPLOADS_DIR, GENERATED_CODE_DIR, OUTPUTS_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── in-memory session store ──
# session_id → { graph, config, state, events_queue, seen_events, last_state, loop }
sessions: dict[str, dict] = {}

# Track background asyncio tasks so we can cancel them on shutdown
_bg_tasks: set[asyncio.Task] = set()


# ─────────────────────── Lifespan ─────────────────────────────

@asynccontextmanager
async def lifespan(_):
    yield
    # Graceful shutdown: signal all SSE streams to close
    logger.info("Shutting down — draining %d active sessions.", len(sessions))
    for session in sessions.values():
        q: asyncio.Queue = session.get("events_queue")
        if q:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
    # Cancel any tracked background tasks
    for task in list(_bg_tasks):
        task.cancel()
    if _bg_tasks:
        await asyncio.gather(*_bg_tasks, return_exceptions=True)


# ─────────────────────── App ──────────────────────────────────

app = FastAPI(title="Agentic AI Data Science", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────── Session helpers ──────────────────────

def _save_session_meta(session_id: str, dataset_path: str, task_type: str, target_column):
    meta = {
        "session_id": session_id,
        "dataset_path": dataset_path,
        "task_type": task_type,
        "target_column": target_column,
    }
    (SESSIONS_DIR / f"{session_id}.json").write_text(json.dumps(meta))


def _load_session_meta(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _reconstruct_session(session_id: str) -> dict | None:
    """Rebuild in-memory session from disk after a server restart."""
    meta = _load_session_meta(session_id)
    if not meta:
        return None

    config = {"configurable": {"thread_id": session_id}}
    try:
        checkpoint = _shared_graph.get_state(config)
        last_state = dict(checkpoint.values) if checkpoint and checkpoint.values else {}
    except Exception:
        last_state = {}

    session = {
        "graph": _shared_graph,
        "config": config,
        "state": create_initial_state(
            meta["dataset_path"], meta["task_type"], meta.get("target_column")
        ),
        "events_queue": asyncio.Queue(),
        "seen_events": len(last_state.get("events", [])),
        "last_state": last_state,
        "dataset_path": meta["dataset_path"],
    }
    sessions[session_id] = session
    logger.info("Reconstructed session %s from disk.", session_id)
    return session


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        session = _reconstruct_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id]


# ─────────────────────── Pipeline runners ─────────────────────

def _make_stream_worker(session_id: str, loop: asyncio.AbstractEventLoop):
    """Return a callable that runs the LangGraph pipeline in a thread pool."""
    session = sessions[session_id]
    graph = session["graph"]
    config = session["config"]
    queue: asyncio.Queue = session["events_queue"]
    state = session["state"]

    def _run():
        try:
            for chunk in graph.stream(state, config=config, stream_mode="values"):
                last_state = chunk
                new_events = last_state.get("events", [])
                already_seen = session.get("seen_events", 0)
                for evt in new_events[already_seen:]:
                    asyncio.run_coroutine_threadsafe(queue.put(evt), loop)
                session["seen_events"] = len(new_events)
                session["last_state"] = last_state
            # Do NOT break early on "awaiting_approval" — interrupt_before must be
            # allowed to fire naturally so LangGraph saves the checkpoint before the
            # generator ends.  Breaking early skips the checkpoint write, leaving
            # stream(None, config) with nothing to resume from.
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(exc), "data": None}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    return _run


def _make_resume_worker(session_id: str, loop: asyncio.AbstractEventLoop, feedback: str = ""):
    """Return a callable that resumes the graph after human approval."""
    session = sessions[session_id]
    graph = session["graph"]
    config = session["config"]
    queue: asyncio.Queue = session["events_queue"]

    def _run():
        try:
            # Push "approved" event to the frontend immediately so the UI updates.
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "approved", "message": "Human approved. Proceeding to ML Engineering.", "data": None}),
                loop,
            )

            # as_node="human_approval" tells LangGraph the human_approval node already
            # ran and produced this state.  The checkpoint's "next" is then computed from
            # edge_after_human_approval(human_approved=True) → "ml_engineering".
            # Without as_node, "next" stays ["human_approval"] and interrupt_before
            # fires again immediately, making stream(None) return with zero chunks.
            graph.update_state(config, {
                "human_approved": True,
                "status": "running",
                "human_feedback": feedback,
            }, as_node="human_approval")
            logger.info("State updated as human_approval node: human_approved=True")

            for chunk in graph.stream(None, config=config, stream_mode="values"):
                last_state = chunk
                new_events = last_state.get("events", [])
                already_seen = session.get("seen_events", 0)
                for evt in new_events[already_seen:]:
                    asyncio.run_coroutine_threadsafe(queue.put(evt), loop)
                session["seen_events"] = len(new_events)
                session["last_state"] = last_state
        except Exception as exc:
            logger.error("Resume error: %s", exc, exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(exc), "data": None}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    return _run


async def _spawn(worker_fn) -> asyncio.Task:
    """Run worker_fn in a thread, track the task for graceful shutdown."""
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, worker_fn)
    # Wrap in a real Task so we can track/cancel it
    tracked = asyncio.ensure_future(task)
    _bg_tasks.add(tracked)
    tracked.add_done_callback(_bg_tasks.discard)
    return tracked


# ─────────────────────── Models ───────────────────────────────

class StartRequest(BaseModel):
    session_id: str
    task_type: str
    target_column: Optional[str] = None


class ApproveRequest(BaseModel):
    feedback: str = ""


# ─────────────────────── Routes ───────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    session_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{session_id}_{file.filename}"

    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Read CSV headers so the frontend can populate a column dropdown
    columns = []
    try:
        with open(dest, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            columns = next(reader, [])
    except Exception:
        pass

    logger.info("Uploaded: %s (session %s)", dest, session_id)
    return {"session_id": session_id, "dataset_path": str(dest), "filename": file.filename, "columns": columns}


@app.post("/start")
async def start_pipeline(body: StartRequest):
    session_id = body.session_id

    matches = list(UPLOADS_DIR.glob(f"{session_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")
    dataset_path = str(matches[0])

    _save_session_meta(session_id, dataset_path, body.task_type, body.target_column)

    config = {"configurable": {"thread_id": session_id}}
    initial_state = create_initial_state(dataset_path, body.task_type, body.target_column)
    queue: asyncio.Queue = asyncio.Queue()

    sessions[session_id] = {
        "graph": _shared_graph,
        "config": config,
        "state": initial_state,
        "events_queue": queue,
        "seen_events": 0,
        "last_state": {},
        "dataset_path": dataset_path,
    }

    loop = asyncio.get_running_loop()
    await _spawn(_make_stream_worker(session_id, loop))
    return {"session_id": session_id, "status": "started"}


@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    session = _get_session(session_id)
    queue: asyncio.Queue = session["events_queue"]

    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to pipeline stream'})}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if event is None:
                        yield f"data: {json.dumps({'type': 'stream_end', 'message': 'Stream ended'})}\n\n"
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'message': 'alive'})}\n\n"
        except asyncio.CancelledError:
            # Normal shutdown — client disconnected or server stopping
            logger.info("SSE stream cancelled for session %s.", session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    session = _get_session(session_id)
    last = session.get("last_state", {})
    return {
        "session_id": session_id,
        "status": last.get("status", "unknown"),
        "task_type": last.get("task_type"),
        "target_column": last.get("target_column"),
        "algorithm_info": last.get("algorithm_info", {}),
        "best_model": last.get("best_model", {}),
        "optimization_iteration": last.get("optimization_iteration", 0),
        "evaluation": last.get("evaluation", {}),
        "has_understanding_output": bool(last.get("understanding_output")),
        "has_analysis_output": bool(last.get("analysis_output")),
        "has_ml_output": bool(last.get("ml_output")),
        "events_count": len(last.get("events", [])),
    }


@app.post("/approve/{session_id}")
async def approve(session_id: str, body: Optional[ApproveRequest] = None):
    session = _get_session(session_id)
    last = session.get("last_state", {})
    if last.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Not awaiting approval. Current status: {last.get('status')}",
        )

    feedback = body.feedback if body else ""
    session["events_queue"] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    await _spawn(_make_resume_worker(session_id, loop, feedback=feedback))
    return {"status": "resumed"}


@app.get("/code/{session_id}/{step}")
async def get_code(session_id: str, step: str):
    _get_session(session_id)
    step_map = {
        "understanding": "step1_understanding.py",
        "analysis": "step2_analysis.py",
        "ml": "step3_ml.py",
    }
    if step not in step_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown step '{step}'. Choose from: {list(step_map)}",
        )
    path = GENERATED_CODE_DIR / step_map[step]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Code not yet generated.")
    return {"step": step, "filename": step_map[step], "code": path.read_text()}


@app.get("/download/{session_id}/model")
async def download_model(session_id: str):
    _get_session(session_id)
    model_path = OUTPUTS_DIR / "model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not yet trained.")
    return FileResponse(str(model_path), filename="model.pkl", media_type="application/octet-stream")


@app.get("/outputs/{session_id}/plots")
async def list_plots(session_id: str):
    _get_session(session_id)
    return {"plots": [p.name for p in OUTPUTS_DIR.glob("*.png")]}


@app.get("/outputs/{session_id}/plot/{filename}")
async def get_plot(session_id: str, filename: str):
    _get_session(session_id)
    path = OUTPUTS_DIR / filename
    if not path.exists() or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(str(path), media_type="image/png")
