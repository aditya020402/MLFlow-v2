"""Executes generated Python scripts and captures their output."""

import json
import re
import subprocess
import sys
import os
import time
import logging
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
    match = re.search(
        r"METRICS_JSON_START\s*(\{.*?\})\s*METRICS_JSON_END",
        stdout,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1))
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


def run_script(script_name: str, timeout: int = 120) -> ExecutionResult:
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


def run_script_with_retry(
    script_name: str,
    fix_callback,
    max_retries: int = 10,
    timeout: int = 120,
) -> ExecutionResult:
    """Run a script; on failure call fix_callback to get fixed code, rewrite, retry."""
    for attempt in range(1, max_retries + 2):
        result = run_script(script_name, timeout=timeout)
        if result.success:
            logger.info("Script passed on attempt %d", attempt)
            return result

        logger.warning("Attempt %d failed. stderr: %s", attempt, result.stderr[:500])

        if attempt > max_retries:
            logger.error("Max retries reached.")
            return result

        logger.info("Requesting fix from agent (attempt %d/%d)...", attempt, max_retries)
        fixed_code = fix_callback(result.stderr, result.stdout, attempt)
        if fixed_code:
            script_path = GENERATED_CODE_DIR / script_name
            script_path.write_text(fixed_code)
            logger.info("Code updated, retrying...")
        else:
            logger.error("Agent returned no fix.")
            return result

    return result
