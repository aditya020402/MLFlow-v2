"""Mutable path configuration for the ada package.

run.py sets these before the pipeline starts so every agent and
the graph write generated scripts and outputs to the user's chosen
output directory instead of a hardcoded backend location.
"""

from pathlib import Path

GENERATED_CODE_DIR: Path = Path("./ada_output/generated_code")
OUTPUTS_DIR: Path        = Path("./ada_output")
UPLOADS_DIR: Path        = Path("./ada_output/data")
