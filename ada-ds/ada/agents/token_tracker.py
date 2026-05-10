"""Session-scoped token usage tracker for all LLM calls in the pipeline."""

import copy
import threading

_lock = threading.Lock()
_usage: dict[str, dict] = {}
_current_session = threading.local()


def set_session(session_id: str) -> None:
    _current_session.id = session_id


def record(agent: str, prompt_tokens: int, completion_tokens: int) -> None:
    session_id = getattr(_current_session, "id", "default")
    with _lock:
        if session_id not in _usage:
            _usage[session_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "by_agent": {},
            }
        s = _usage[session_id]
        s["input_tokens"]  += prompt_tokens
        s["output_tokens"] += completion_tokens
        s["total_tokens"]  += prompt_tokens + completion_tokens

        if agent not in s["by_agent"]:
            s["by_agent"][agent] = {"input": 0, "output": 0, "total": 0, "calls": 0}
        a = s["by_agent"][agent]
        a["input"]  += prompt_tokens
        a["output"] += completion_tokens
        a["total"]  += prompt_tokens + completion_tokens
        a["calls"]  += 1


def get(session_id: str) -> dict:
    with _lock:
        return copy.deepcopy(_usage.get(session_id, {}))
