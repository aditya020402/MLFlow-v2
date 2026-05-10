"""Executes generated Python scripts and captures their output."""

import json
import re
import subprocess
import sys
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [RUNNER] %(message)s")
logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


@dataclass
class ExecutionResult:
    """Result from executing a generated Python script."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    script_path: str


def parse_metrics_from_output(stdout: str) -> dict:
    """Extract JSON metrics enclosed in METRICS_JSON_START / METRICS_JSON_END markers.

    Falls back to the last JSON object in stdout if markers are absent.
    """
    start = stdout.find("METRICS_JSON_START")
    end   = stdout.find("METRICS_JSON_END")
    if start != -1 and end != -1 and end > start:
        raw = stdout[start + len("METRICS_JSON_START"):end].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Fallback: scan lines in reverse for a JSON object
    for line in reversed(stdout.strip().split("\n")):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return {}


def get_primary_score(metrics: dict) -> float:
    """Return a single comparable float score from a metrics dict."""
    for key in ("f1", "accuracy", "r2_score", "silhouette_score"):
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def run_script(script_name: str, timeout: int = 300) -> ExecutionResult:
    """Execute a Python script from generated_code/ and capture all output."""
    script_path = GENERATED_CODE_DIR / script_name
    if not script_path.exists():
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Script not found: {script_path}",
            exit_code=-1,
            duration_seconds=0.0,
            script_path=str(script_path),
        )

    logger.info("Executing: %s", script_path)
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(GENERATED_CODE_DIR),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                # libomp is keg-only on macOS (brew install libomp); xgboost and
                # lightgbm need this path to load their native OpenMP runtime.
                "DYLD_LIBRARY_PATH": "/opt/homebrew/opt/libomp/lib:"
                    + os.environ.get("DYLD_LIBRARY_PATH", ""),
            },
            check=False,
        )
        duration = time.time() - start
        success = result.returncode == 0

        if success:
            logger.info("Script succeeded in %.2fs", duration)
        else:
            logger.warning("Script failed (exit %d) in %.2fs", result.returncode, duration)

        return ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            duration_seconds=duration,
            script_path=str(script_path),
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        logger.error("Script timed out after %ds", timeout)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            exit_code=-2,
            duration_seconds=duration,
            script_path=str(script_path),
        )
    except OSError as exc:
        duration = time.time() - start
        logger.error("Runner OS error: %s", exc)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(exc),
            exit_code=-3,
            duration_seconds=duration,
            script_path=str(script_path),
        )


def _append_log(log_path: Path, entry: dict) -> None:
    """Append one execution entry to the session's log.json (creates if missing)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(entry)
        log_path.write_text(json.dumps(existing, indent=2, default=str))
    except Exception as exc:
        logger.warning("Could not write log.json: %s", exc)


def run_script_with_retry(
    script_name: str,
    fix_callback,
    max_retries: int = 10,
    timeout: int = 300,
    log_path: Path | None = None,
) -> ExecutionResult:
    """Run a script; on failure call fix_callback to get fixed code, rewrite, retry."""
    for attempt in range(1, max_retries + 2):
        result = run_script(script_name, timeout=timeout)

        if log_path:
            metrics = parse_metrics_from_output(result.stdout) if result.success else {}
            _append_log(log_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": script_name,
                "attempt": attempt,
                "success": result.success,
                "exit_code": result.exit_code,
                "duration_seconds": round(result.duration_seconds, 2),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": metrics,
            })

        if result.success:
            logger.info("Script passed on attempt %d", attempt)
            return result

        # The generated except block prints {"error": ...} to stdout (not stderr).
        # Extract it so the log shows the real error and the fix agent can read it.
        effective_stderr = result.stderr
        if not effective_stderr.strip() and result.stdout.strip():
            try:
                parsed = json.loads(result.stdout.strip())
                if "error" in parsed:
                    effective_stderr = f"[Script exception caught by except block]\n{parsed['error']}"
            except Exception:
                # stdout isn't JSON; show the first 500 chars as-is
                effective_stderr = result.stdout[:500]

        logger.warning("Attempt %d failed. error: %s", attempt, effective_stderr[:500])

        if attempt > max_retries:
            logger.error("Max retries reached.")
            return result

        logger.info("Requesting fix from agent (attempt %d/%d)...", attempt, max_retries)
        fixed_code = fix_callback(effective_stderr, result.stdout, attempt)
        if fixed_code:
            script_path = GENERATED_CODE_DIR / script_name
            script_path.write_text(fixed_code, encoding="utf-8")
            logger.info("Code updated, retrying...")
        else:
            logger.error("Agent returned no fix.")
            return result

    return result
