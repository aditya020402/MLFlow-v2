"""Executes generated Python scripts and captures their output."""

import json
import subprocess
import sys
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from ada import paths as _paths


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    script_path: str


def parse_metrics_from_output(stdout: str) -> dict:
    start = stdout.find("METRICS_JSON_START")
    end   = stdout.find("METRICS_JSON_END")
    if start != -1 and end != -1 and end > start:
        raw = stdout[start + len("METRICS_JSON_START"):end].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    for line in reversed(stdout.strip().split("\n")):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return {}


def get_primary_score(metrics: dict) -> float:
    for key in ("f1", "accuracy", "r2_score", "silhouette_score"):
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def run_script(script_name: str, timeout: int = 300) -> ExecutionResult:
    script_path = _paths.GENERATED_CODE_DIR / script_name
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
            cwd=str(_paths.GENERATED_CODE_DIR),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
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
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(entry)
        log_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write log.json: %s", exc)


def run_script_with_retry(
    script_name: str,
    fix_callback,
    max_retries: int = 10,
    timeout: int = 300,
    log_path: Path | None = None,
) -> ExecutionResult:
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

        effective_stderr = result.stderr
        if not effective_stderr.strip() and result.stdout.strip():
            try:
                parsed = json.loads(result.stdout.strip())
                if "error" in parsed:
                    effective_stderr = f"[Script exception caught by except block]\n{parsed['error']}"
            except Exception:
                effective_stderr = result.stdout[:500]

        logger.warning("Attempt %d failed. error: %s", attempt, effective_stderr[:500])

        if attempt > max_retries:
            logger.error("Max retries reached.")
            return result

        logger.info("Requesting fix from agent (attempt %d/%d)...", attempt, max_retries)
        fixed_code = fix_callback(effective_stderr, result.stdout, attempt)
        if fixed_code:
            script_path = _paths.GENERATED_CODE_DIR / script_name
            script_path.write_text(fixed_code, encoding="utf-8")
            logger.info("Code updated, retrying...")
        else:
            logger.error("Agent returned no fix.")
            return result

    return result
