import logging
import json
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

GENERATED_CODE_DIR = Path(__file__).parent.parent / "generated_code"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

client = OpenAI()
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert Data Analyst Agent. Your ONLY job is to write clean, executable Python code.

Rules:
- Output ONLY valid Python code. No markdown, no backticks, no explanations.
- The code must be completely self-contained and runnable.
- Use pandas, numpy, scipy.stats — all available.
- Print a structured JSON summary to stdout at the end: print(json.dumps(analysis, indent=2, default=str))
- The analysis dict must include: correlations, outlier_counts, feature_insights, top_features, recommendations, preprocessing_decisions.
- Save any plots as PNG files in the outputs/ directory using matplotlib with Agg backend (no display).
- Choose the most appropriate imputation strategy based on the data's distribution and skewness.
"""

FIX_SYSTEM_PROMPT = """You are an expert Python debugger. Fix the provided code based on the error.
Output ONLY the complete fixed Python code. No explanations, no markdown, no backticks."""


def generate_analysis_code(dataset_path: str, understanding_output: str, task_type: str, target_column: str = None) -> str:
    outputs_dir = str(OUTPUTS_DIR)

    prompt = f"""Write Python code to perform exploratory data analysis on the dataset at: {dataset_path}

Task type: {task_type}
{"Target column: " + target_column if target_column else ""}

Data Understanding Summary (from previous step):
{understanding_output}

The code must:
1. Load the full dataset using pandas (read_csv)
2. Handle missing values intelligently:
   - Drop columns where >50% of values are missing
   - For remaining columns, choose an appropriate imputation strategy based on the data:
     * Numeric columns: use median for skewed distributions, mean for symmetric ones
     * Categorical columns: use mode
   - Document each imputation decision in the preprocessing_decisions dict
3. Compute Pearson correlation matrix for numeric columns
4. Detect outliers using IQR method (values outside Q1-1.5*IQR to Q3+1.5*IQR)
5. {"Analyze feature-target relationships for target column: " + target_column if target_column else "Analyze feature clusters and variance"}
6. Generate and SAVE (do NOT show/display) the following plots to {outputs_dir}/:
   - correlation_heatmap.png  (seaborn heatmap or matplotlib)
   - distributions.png (histograms for top 6 numeric features)
7. Build an analysis dict and print it: print(json.dumps(analysis, indent=2, default=str))

The analysis dict structure:
{{
  "correlations": {{"col1_col2": 0.85, ...}},
  "outlier_counts": {{"col": count, ...}},
  "feature_insights": ["insight 1", "insight 2", ...],
  "top_features": ["col1", "col2"],
  "recommendations": ["rec 1", "rec 2"],
  "preprocessing_decisions": {{"col": "strategy used e.g. median imputation (skewed)", ...}},
  "plots_saved": ["correlation_heatmap.png", "distributions.png"]
}}

IMPORTANT: Use matplotlib with Agg backend:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Output ONLY the Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    code = response.choices[0].message.content.strip()
    code = _strip_markdown(code)
    logger.info("Data Analyst Agent generated code.")
    return code


def fix_analysis_code(current_code: str, stderr: str, stdout: str, attempt: int) -> str:
    logger.info(f"Data Analyst Agent fixing code (attempt {attempt})...")

    prompt = f"""The following Python code failed to execute.

ERROR:
{stderr}

STDOUT (may be partial):
{stdout}

CURRENT CODE:
{current_code}

Fix the code so it runs correctly. Output ONLY the complete fixed Python code."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    fixed = response.choices[0].message.content.strip()
    fixed = _strip_markdown(fixed)
    logger.info("Data Analyst Agent returned fix.")
    return fixed


def make_fix_callback(current_code_path: Path):
    def callback(stderr: str, stdout: str, attempt: int) -> str:
        current_code = current_code_path.read_text()
        return fix_analysis_code(current_code, stderr, stdout, attempt)
    return callback


def _strip_markdown(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def run(dataset_path: str, understanding_output: str, task_type: str, target_column: str = None) -> dict:
    code = generate_analysis_code(dataset_path, understanding_output, task_type, target_column)
    script_path = GENERATED_CODE_DIR / "step2_analysis.py"
    script_path.write_text(code)
    logger.info(f"Written: {script_path}")
    return {
        "script_name": "step2_analysis.py",
        "script_path": str(script_path),
        "code": code,
        "fix_callback": make_fix_callback(script_path),
    }
