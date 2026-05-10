# Ada — The Digital Data Scientist
## Complete Usage Guide

---

## Table of Contents

1. [What is Ada?](#what-is-ada)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Instructions File Format](#instructions-file-format)
5. [Usage: pip / Python API & CLI](#usage-pip--python-api--cli)
6. [Usage: Web UI](#usage-web-ui)
7. [Usage: GitLab Integration](#usage-gitlab-integration)
8. [Pipeline Architecture Diagram](#pipeline-architecture-diagram)
9. [Output Structure](#output-structure)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## What is Ada?

Ada is an **end-to-end AutoML agent** that autonomously handles the complete data science workflow. You give it a CSV dataset and a plain-English instructions file — Ada figures out the rest.

**Supported task types:**
- Supervised **Classification**
- Supervised **Regression**
- Unsupervised **Clustering**

**Ada runs 9 specialized AI agents in sequence:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADA AGENT PIPELINE                               │
│                                                                         │
│  [1a] Data Understanding  →  Analyze columns, types, missing values     │
│  [1b] Task Profiler       →  Task-specific analysis (imbalance, etc.)   │
│  [2]  Data Analyst        →  Clean data, generate analysis report        │
│       ↓  (Human Approval Gate — prompts for optional feedback)          │
│  [3]  ML Engineer         →  Train baseline model                        │
│  [4]  Optimizer           →  Tune hyperparameters / try new algorithms  │
│  [5]  Evaluator           →  Score model, decide pass/retry              │
│  [6]  LLM Client          →  Shared GPT-4o-mini interface               │
│  [7]  Token Tracker       →  Monitor API costs                           │
│  [8]  Testing Agent       →  Validate final model on held-out data       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Three ways to use Ada:**

```
┌───────────────────────────────────────────────────────────┐
│                   HOW TO USE ADA                          │
│                                                           │
│  ① pip install   →  ada CLI or Python API                 │
│                     (terminal / scripts / notebooks)       │
│                                                           │
│  ② Web UI        →  Browser interface                     │
│                     (upload CSV, watch live progress)      │
│                                                           │
│  ③ GitLab        →  Issue-driven automation               │
│                     (create issue → get model back)        │
└───────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install
cd MLFlow-v2/ada-ds
pip install -e .

# 2. Set your OpenAI key
echo "OPENAI_API_KEY=sk-proj-..." > .env

# 3. Run on your data
ada run --dataset my_data.csv --instructions instructions.md
```

---

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.10  |
| pip         | latest  |
| OpenAI API key | required |

### Option A — Editable install (development)

```bash
cd ada-ds
pip install -e .
```

### Option B — Standard install (from local wheel)

```bash
cd ada-ds
pip install .
```

### Option C — Web UI + Backend (full stack)

```bash
# Backend dependencies
pip install -e ada-ds/

# Frontend dependencies
cd frontend
npm install
```

### Verify installation

```bash
ada --help
# Expected output:
# Usage: ada [OPTIONS] COMMAND [ARGS]...
# Commands:
#   run   Run the Ada ML pipeline
```

---

## Instructions File Format

The instructions file (`.md`) is **how you communicate with Ada**. It controls every aspect of the pipeline.

### Full Template

```markdown
## Task Type
classification

## Target Variable
`target_column_name`

## Business Context
Describe the problem in plain English. Ada uses this to understand
what "good" looks like for your use case.

## Data Notes
- Column X contains erroneous zeros, treat as missing
- Date column is MM/DD/YYYY format
- Column Y is a proxy for the target, exclude it

## ML Requirements
- Prefer recall over precision
- Try XGBoost and RandomForest
- Feature importance is important

## Evaluation Criteria
primary metric: f1

## Validation
yes
```

### Field Reference

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `Task Type` | **Yes** | `classification` \| `regression` \| `clustering` | Type of ML task |
| `Target Variable` | Yes (class/regress) | Column name in backticks | The column to predict |
| `Business Context` | Recommended | Free text | Helps Ada understand the domain |
| `Data Notes` | No | Bullet list | Domain-specific data quirks |
| `ML Requirements` | No | Bullet list | Algorithm preferences, priorities |
| `Evaluation Criteria` | No | `primary metric: <name>` | `f1`, `accuracy`, `r2`, `silhouette` |
| `Validation` | No | `yes` \| `no` | Hold out 5% of data for final validation |

### Task Type Examples

**Classification:**
```markdown
## Task Type
classification

## Target Variable
`churn`

## Evaluation Criteria
primary metric: f1
```

**Regression:**
```markdown
## Task Type
regression

## Target Variable
`house_price`

## Evaluation Criteria
primary metric: r2
```

**Clustering:**
```markdown
## Task Type
clustering

## Evaluation Criteria
primary metric: silhouette
```

---

## Usage: pip / Python API & CLI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIP USAGE — COMPLETE DIAGRAM                            │
│                                                                             │
│  SETUP                                                                      │
│  ─────                                                                      │
│  pip install -e ada-ds/                                                     │
│  echo "OPENAI_API_KEY=sk-..." > .env                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         TWO INTERFACES                               │  │
│  │                                                                      │  │
│  │  ① CLI COMMAND                    ② PYTHON API                      │  │
│  │  ─────────────                    ────────────                      │  │
│  │  ada run \                        from ada import run_pipeline       │  │
│  │    --dataset  data.csv \                                             │  │
│  │    --instructions inst.md \       result = run_pipeline(            │  │
│  │    --output   ./output \            dataset="data.csv",             │  │
│  │    --env      .env \               instructions="inst.md",          │  │
│  │    --quiet                          output_dir="./output",          │  │
│  │                                     env_file=".env",                │  │
│  │  FLAGS:                             verbose=True                    │  │
│  │  --dataset  / -d  (required)      )                                 │  │
│  │  --instructions / -i (required)                                     │  │
│  │  --output   / -o  (./ada_output)  RETURNS dict:                     │  │
│  │  --env      / -e  (.env)          {                                 │  │
│  │  --quiet    / -q  (suppress logs)   "status": "completed",          │  │
│  │                                     "model_path": "output/.../      │  │
│  │                                      best_XGB_f1_0.8934.pkl",       │  │
│  │                                     "session_dir": "output/abc123", │  │
│  │                                     "summary_path": "...json",      │  │
│  │                                     "token_usage": {...},           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  PIPELINE EXECUTION (both interfaces run the same flow)                     │
│  ──────────────────                                                         │
│                                                                             │
│   your data.csv ──┐                                                         │
│   instructions.md ┤                                                         │
│   .env (API key)  ┘                                                         │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│   │  Step 1a    │ → │  Step 1b    │ → │   Step 2    │                      │
│   │  Data       │   │  Task       │   │   Data      │                      │
│   │  Understand │   │  Profiler   │   │   Analyst   │                      │
│   └─────────────┘   └─────────────┘   └─────────────┘                      │
│                                              │                              │
│                                    [Human Approval Gate]                    │
│                                    Ada pauses here and                      │
│                                    shows analysis summary.                  │
│                                    Press Enter to continue                  │
│                                    (or type optional feedback)              │
│                                              │                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│   │  Evaluator  │ ← │  Optimizer  │ ← │  ML Engineer│                      │
│   │  pass/retry │   │  tune/new   │   │  baseline   │                      │
│   └─────────────┘   └─────────────┘   └─────────────┘                      │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────────────────────────────────────────────────┐                │
│   │                   OUTPUT FILES                        │                │
│   │                                                       │                │
│   │  ./output/                                            │                │
│   │  └── abc12345/           ← unique session folder      │                │
│   │      ├── best_XGB_f1_0.8934.pkl    ← best model       │                │
│   │      ├── best_XGB_f1_0.8934_       ← feature list     │                │
│   │      │   features.pkl                                  │                │
│   │      ├── summary.json              ← full run report   │                │
│   │      ├── optimization_history.json                     │                │
│   │      ├── generated_code/                               │                │
│   │      │   ├── step1a_general.py                         │                │
│   │      │   ├── step1b_profiling.py                       │                │
│   │      │   ├── step2_analysis.py                         │                │
│   │      │   ├── step3_ml.py                               │                │
│   │      │   └── step3_ml_iter1.py                         │                │
│   │      └── data/                                         │                │
│   │          └── your_data.csv                             │                │
│   └───────────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Console Output Icons

While running, Ada prints clean one-line status updates:

```
[...] Thinking / generating code
[-->] Code file written
[>>>] Running a script
[ OK] Script succeeded
[ERR] Script failed (will retry automatically)
[ALG] Algorithm selected
[WHY] Rationale — why this algorithm was chosen, what hyperparameters
[NEW] New best model found
[OPT] Optimization complete
[EVL] Evaluation verdict
[VAL] Held-out validation results
[TOK] Token usage summary
[DONE] Pipeline finished
```

### CLI Examples

```bash
# Minimal — classification
ada run -d bank_train.csv -i instructions.md

# With custom output directory
ada run -d data.csv -i instructions.md -o ./my_results

# With custom .env file
ada run -d data.csv -i instructions.md -e /path/to/.env

# Suppress extra log output
ada run -d data.csv -i instructions.md --quiet
```

### Python API Example

```python
from ada import run_pipeline

result = run_pipeline(
    dataset="bank_train.csv",
    instructions="instructions.md",
    output_dir="./results",
    env_file=".env",
    verbose=True
)

# Access results
print(result["status"])          # "completed"
print(result["model_path"])      # e.g. "./results/abc12345/best_XGB_f1_0.8934.pkl"
print(result["session_dir"])     # "./results/abc12345"
print(result["evaluation"])      # {"verdict": "pass", "score": 0.89, ...}
print(result["token_usage"])     # {"total_prompt": 50000, ...}

# Load the trained model using the path from the result
import pickle
with open(result["model_path"], "rb") as f:
    model = pickle.load(f)
```

### Jupyter Notebook Example

```python
# Cell 1: Install
# !pip install -e /path/to/ada-ds

# Cell 2: Set API key inline (alternative to .env)
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["MAX_OPTIMIZATION_LOOPS"] = "2"

# Cell 3: Run
from ada import run_pipeline
result = run_pipeline(
    dataset="data.csv",
    instructions="instructions.md",
    output_dir="./output"
)

# Cell 4: Inspect
import json
with open(result["summary_path"]) as f:
    summary = json.load(f)
print(json.dumps(summary["best_model"], indent=2))
```

### Human Approval Gate (CLI/API)

After data analysis completes, Ada **pauses and shows you a summary** before starting ML modeling:

```
============================================================
DATA ANALYSIS COMPLETE — Review before ML modeling
============================================================
  Original shape : (10000, 15)
  Cleaned shape  : (9847, 15)

  Recommendations:
    - 3 columns with >20% missing values were imputed
    - High cardinality column 'id' was dropped

  Suggested models:
    - XGBoostClassifier
    - RandomForestClassifier

  Press Enter to proceed to ML modeling.
  You may also type optional feedback/instructions and press Enter.
------------------------------------------------------------
  Your feedback (or just Enter): try SVM first, prioritize recall
============================================================
```

Any feedback you type here is passed directly to the ML agent and influences algorithm selection and optimization priorities.

---

## Usage: Web UI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WEB UI — COMPLETE USAGE DIAGRAM                        │
│                                                                             │
│  SETUP (one time)                                                           │
│  ─────────────────                                                          │
│                                                                             │
│  Terminal 1 — Backend:          Terminal 2 — Frontend:                      │
│  ┌────────────────────────┐     ┌────────────────────────┐                  │
│  │ cd MLFlow-v2           │     │ cd MLFlow-v2/frontend  │                  │
│  │ pip install -e ada-ds/ │     │ npm install            │                  │
│  │ python backend/        │     │ npm run dev            │                  │
│  │        run_server.py   │     │                        │                  │
│  │                        │     │ → http://localhost:5173 │                  │
│  │ → http://localhost:8000│     └────────────────────────┘                  │
│  └────────────────────────┘                                                 │
│                                                                             │
│  STEP-BY-STEP UI FLOW                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SCREEN 1: Upload Screen (http://localhost:5173)                     │  │
│  │                                                                      │  │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐           │  │
│  │  │   Drop your CSV here    │  │  Drop instructions.md   │           │  │
│  │  │   or click to browse    │  │  or click to browse     │           │  │
│  │  │                         │  │                         │           │  │
│  │  │   bank_train.csv ✓      │  │   instructions.md ✓     │           │  │
│  │  └─────────────────────────┘  └─────────────────────────┘           │  │
│  │                                                                      │  │
│  │                    [ Start Pipeline ]                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                │                                                            │
│                ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SCREEN 2: Pipeline Dashboard (live progress)                        │  │
│  │                                                                      │  │
│  │  Tabs: [Understanding] [Analysis] [ML Model] [Results]               │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────┐             │  │
│  │  │  AGENT CONSOLE (live streaming output)             │             │  │
│  │  │                                                    │             │  │
│  │  │  [...] Analyzing column types...                   │             │  │
│  │  │  [ OK] General understanding complete.             │             │  │
│  │  │  [ALG] Baseline algorithm: XGBoostClassifier       │             │  │
│  │  │  [WHY] High cardinality + imbalanced target...     │             │  │
│  │  │  [NEW] New best: XGBoost (f1=0.8934)               │             │  │
│  │  │  ...                                               │             │  │
│  │  └────────────────────────────────────────────────────┘             │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────┐             │  │
│  │  │  CODE VIEWER (inspect generated Python)            │             │  │
│  │  │  [understanding] [analysis] [ml]                   │             │  │
│  │  │                                                    │             │  │
│  │  │  # step1a_general.py (auto-generated)              │             │  │
│  │  │  import pandas as pd                               │             │  │
│  │  │  ...                                               │             │  │
│  │  └────────────────────────────────────────────────────┘             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                │                                                            │
│                ▼  (after Step 2 completes)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  HUMAN APPROVAL GATE (optional pause point)                          │  │
│  │                                                                      │  │
│  │  Ada has completed data analysis. Review the results above.          │  │
│  │                                                                      │  │
│  │  Optional feedback:                                                  │  │
│  │  ┌────────────────────────────────────────────────────┐             │  │
│  │  │ "Focus more on recall, the class imbalance needs   │             │  │
│  │  │  to be addressed with SMOTE"                       │             │  │
│  │  └────────────────────────────────────────────────────┘             │  │
│  │                                                                      │  │
│  │  [ Approve & Continue ]    [ Provide Feedback ]                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                │                                                            │
│                ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  SCREEN 3: Results Dashboard                                         │  │
│  │                                                                      │  │
│  │  Best Model: XGBoostClassifier                                       │  │
│  │  ┌──────────────┬───────────┐                                        │  │
│  │  │ Accuracy     │  0.9012   │                                        │  │
│  │  │ F1 Score     │  0.8934   │                                        │  │
│  │  │ Precision    │  0.8865   │                                        │  │
│  │  │ Recall       │  0.8980   │                                        │  │
│  │  └──────────────┴───────────┘                                        │  │
│  │                                                                      │  │
│  │  Evaluation: ✓ PASS                                                  │  │
│  │  Strengths: [high recall, robust to noise, ...]                      │  │
│  │  Weaknesses: [slight overfitting, ...]                               │  │
│  │                                                                      │  │
│  │  [ Download model ]    [ View Analysis Plots ]                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Backend API Reference

The React UI talks to the FastAPI backend at `http://localhost:8000`. All endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/upload` | Upload CSV dataset → returns `session_id` |
| `POST` | `/start` | Start pipeline with `session_id` + instructions |
| `GET` | `/stream/{session_id}` | SSE stream — live agent output (subscribe to this) |
| `GET` | `/state/{session_id}` | Current pipeline state snapshot |
| `POST` | `/approve/{session_id}` | Inject human approval + optional feedback |
| `GET` | `/code/{session_id}/{step}` | Get generated code (`understanding`\|`analysis`\|`ml`) |
| `GET` | `/download/{session_id}/model` | Download the best model `.pkl` |
| `GET` | `/health` | Health check |

### Starting Servers

```bash
# Backend (FastAPI on port 8000)
cd MLFlow-v2
python backend/run_server.py

# Frontend (Vite dev server on port 5173)
cd MLFlow-v2/frontend
npm run dev
```

### Environment for Web UI

Create `backend/.env`:
```
OPENAI_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o-mini
MAX_RETRIES=10
MAX_OPTIMIZATION_LOOPS=1
MAX_TUNE_ITERATIONS=1
```

---

## Usage: GitLab Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GITLAB INTEGRATION — COMPLETE DIAGRAM                    │
│                                                                             │
│  WHAT IT DOES                                                               │
│  ────────────                                                               │
│  Ada monitors your GitLab project for issues labeled "ml-ready".           │
│  You attach a CSV and write ML instructions in the issue body.              │
│  Ada trains a model and posts results + model back to the issue.            │
│                                                                             │
│  SETUP (one time)                                                           │
│  ─────────────────                                                          │
│                                                                             │
│  1. Configure backend/gitlab_backend/.env:                                  │
│     OPENAI_API_KEY=sk-proj-...                                              │
│     GITLAB_URL=https://gitlab.com                                           │
│     GITLAB_TOKEN=glpat-xxxxxxxxxxxx    ← Personal Access Token             │
│                                          (needs api + read_api scopes)     │
│     POLL_INTERVAL=60                   ← seconds between polls             │
│                                                                             │
│  2. Start the GitLab backend service:                                       │
│     cd MLFlow-v2                                                            │
│     python backend/gitlab_backend/main.py                                  │
│     → Service runs on http://localhost:8001                                │
│                                                                             │
│  3. Register your GitLab project via API:                                  │
│     curl -X POST http://localhost:8001/register \                           │
│       -H "Content-Type: application/json" \                                │
│       -d '{"project_id": "myorg/myrepo"}'                                  │
│                                                                             │
│  THE WORKFLOW                                                               │
│  ─────────────                                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GITLAB ISSUE (you create this)                                     │   │
│  │                                                                     │   │
│  │  Title: "Train churn prediction model"                              │   │
│  │  Labels: ml-ready                          ← TRIGGER LABEL          │   │
│  │                                                                     │   │
│  │  Body:                                                              │   │
│  │  ─────                                                              │   │
│  │  ## Task Type                                                       │   │
│  │  classification                                                     │   │
│  │                                                                     │   │
│  │  ## Target Variable                                                 │   │
│  │  `churn`                                                            │   │
│  │                                                                     │   │
│  │  ## Business Context                                                │   │
│  │  Predict customer churn for telecom customers.                      │   │
│  │                                                                     │   │
│  │  ## ML Requirements                                                 │   │
│  │  - Maximize recall (false negatives are costly)                     │   │
│  │  - Try XGBoost                                                      │   │
│  │                                                                     │   │
│  │  ## Evaluation Criteria                                             │   │
│  │  primary metric: f1                                                 │   │
│  │                                                                     │   │
│  │  Attachments: [data.csv] (/uploads/abc123/data.csv)  ← attach CSV  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                │                                                            │
│                ▼  Ada polls every POLL_INTERVAL seconds                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ADA DETECTS ISSUE                                                  │   │
│  │  • Finds issue with label "ml-ready"                                │   │
│  │  • Downloads CSV attachment                                         │   │
│  │  • Parses issue body as instructions                                │   │
│  │  • Removes "ml-ready", adds "ml-processing" label                  │   │
│  │  • Posts comment: "Ada started pipeline — session: abc123"          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                │                                                            │
│                ▼  Pipeline runs (same as pip/UI mode)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PROGRESS COMMENTS (Ada posts automatically)                        │   │
│  │                                                                     │   │
│  │  Comment 1: "Step 1 complete: 12 features, no nulls found"          │   │
│  │  Comment 2: "Step 2 complete: cleaned dataset ready"                │   │
│  │  Comment 3: "Baseline model: RandomForest — F1: 0.72"               │   │
│  │  Comment 4: "Optimization iter 1: XGBoost — F1: 0.89 ✓"            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                │                                                            │
│                ▼  On completion                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ADA POSTS FINAL RESULTS TO ISSUE                                   │   │
│  │                                                                     │   │
│  │  Comment: "Pipeline complete ✓"                                     │   │
│  │  ┌─────────────────────────────────────────────┐                   │   │
│  │  │  Best Model: XGBoostClassifier               │                   │   │
│  │  │  F1 Score: 0.8934                            │                   │   │
│  │  │  Accuracy: 0.9012                            │                   │   │
│  │  │  Recall: 0.8980                              │                   │   │
│  │  │  Verdict: PASS                               │                   │   │
│  │  └─────────────────────────────────────────────┘                   │   │
│  │                                                                     │   │
│  │  Attachments uploaded:                                              │   │
│  │    • best_XGBoostClassifier_f1_0.8934.pkl  ← best model            │   │
│  │    • summary.json                          ← full run report        │   │
│  │                                                                     │   │
│  │  Label: "ml-complete" added                                         │   │
│  │  Issue: closed                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ERROR HANDLING                                                             │
│  ──────────────                                                             │
│  If the pipeline fails:                                                     │
│  • Label changes from "ml-processing" → "ml-failed"                        │
│  • Ada posts error details as a comment                                     │
│  • Issue remains open for you to investigate                                │
│                                                                             │
│  LABEL LIFECYCLE                                                            │
│  ────────────────                                                           │
│  ml-ready → ml-processing → ml-complete  (success)                         │
│                           → ml-failed    (error)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GitLab Issue Body Format

```markdown
## Task Type
classification

## Target Variable
`column_name`

## Business Context
Free text description of the ML problem.

## Data Notes
- Any data quirks, domain knowledge
- Column X is a proxy for the target, exclude it

## ML Requirements
- prefer recall over precision
- try XGBoost

## Evaluation Criteria
primary metric: f1

## Validation
yes
```

Attach your CSV file directly to the issue, then add the label `ml-ready`.

### GitLab Backend API

The service runs on port 8001:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/register` | Register a GitLab project to monitor |
| `GET` | `/projects` | List all registered projects |
| `DELETE` | `/projects/{id}` | Unregister a project |
| `GET` | `/health` | Health check |

---

## Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ADA FULL PIPELINE ARCHITECTURE                             │
│                                                                             │
│  INPUT                                                                      │
│  ─────                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │  dataset    │    │  instructions.md  │    │  .env                      │ │
│  │  (CSV)      │    │  task_type        │    │  OPENAI_API_KEY            │ │
│  │             │    │  target_column    │    │  MAX_OPTIMIZATION_LOOPS=3  │ │
│  │  rows: any  │    │  business_context │    │  MAX_TUNE_ITERATIONS=2     │ │
│  │  cols: any  │    │  ml_requirements  │    │  MAX_RETRIES=10            │ │
│  └─────────────┘    └──────────────────┘    └────────────────────────────┘ │
│         │                   │                           │                   │
│         └───────────────────┴───────────────────────────┘                  │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     LANGGRAPH ORCHESTRATOR                           │  │
│  │              (in-process execution — no persistence)                 │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  NODE 1: data_understanding                                    │ │  │
│  │  │  • Generates step1a_general.py                                 │ │  │
│  │  │  • Executes it → column types, nulls, outliers, correlations   │ │  │
│  │  │  • Output: understanding_output (JSON stats)                   │ │  │
│  │  │  • Retry limit: MAX_RETRIES                                    │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │                             ▼                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  NODE 2: task_profiler                                         │ │  │
│  │  │  • Generates step1b_profiling.py                               │ │  │
│  │  │  • Classification: imbalance, separability, confusion baseline │ │  │
│  │  │  • Regression: target distribution, heteroscedasticity         │ │  │
│  │  │  • Clustering: variance, PCA, k-range suggestion               │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │                             ▼                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  NODE 3: data_analyst                                          │ │  │
│  │  │  • Generates step2_analysis.py                                 │ │  │
│  │  │  • Cleans data (impute, encode, outliers, scaling)             │ │  │
│  │  │  • Produces cleaned CSV + analysis_report.json                 │ │  │
│  │  │  • Generates correlation plots, distribution charts            │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │                             ▼                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  HUMAN APPROVAL GATE                                           │ │  │
│  │  │  • CLI/API: Ada pauses, prints analysis summary, waits for     │ │  │
│  │  │    Enter key. Optional typed feedback is passed to ML agents.  │ │  │
│  │  │  • Web UI: waits for user click + optional feedback text       │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │                             ▼                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  NODE 4: ml_engineer                                           │ │  │
│  │  │  • Generates step3_ml.py (iteration 0 / baseline)              │ │  │
│  │  │  • LLM selects algorithm based on task + data profile          │ │  │
│  │  │  • Respects algorithm preferences from user feedback           │ │  │
│  │  │  • Trains model, saves model.pkl + _features.pkl               │ │  │
│  │  │  • Prints [WHY] rationale for algorithm choice                 │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │                             ▼                                        │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │  NODE 5: track_metrics                                         │ │  │
│  │  │  • Parses metrics JSON from script output                      │ │  │
│  │  │  • Updates best_model if score improved                        │ │  │
│  │  │  • Appends to optimization_history                             │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                             │                                        │  │
│  │              ┌──────────────┴──────────────┐                        │  │
│  │              ▼                             ▼                         │  │
│  │     [more iterations left]         [budget exhausted]                │  │
│  │              │                             │                         │  │
│  │              ▼                             ▼                         │  │
│  │  ┌──────────────────────┐     ┌──────────────────────────────────┐  │  │
│  │  │  NODE 6: optimizer   │     │  NODE 7: finalize_best           │  │  │
│  │  │                      │     │  • Copies best model to          │  │  │
│  │  │  Strategy A:         │     │    best_{Algo}_{metric}_{score}  │  │  │
│  │  │  tune_best           │     │    .pkl in session folder        │  │  │
│  │  │  → RandomizedSearchCV│     │  • Copies matching _features.pkl │  │  │
│  │  │    on best model     │     │  • Writes optimization_history   │  │  │
│  │  │                      │     └──────────────────────────────────┘  │  │
│  │  │  Strategy B:         │                  │                         │  │
│  │  │  new_algorithm       │                  ▼                         │  │
│  │  │  → pick untried algo │     ┌──────────────────────────────────┐  │  │
│  │  │    + quick search    │     │  NODE 8: evaluator               │  │  │
│  │  │                      │     │  • LLM scores model quality      │  │  │
│  │  │  Generates:          │     │  • Verdict: pass / retry         │  │  │
│  │  │  step3_ml_iter1.py   │     │  • Pass criteria:                │  │  │
│  │  │  step3_ml_iter2.py   │     │    Classification: F1 ≥ 0.70    │  │  │
│  │  │  ...                 │     │    Regression: R² ≥ 0.60        │  │  │
│  │  │  Prints [WHY] for    │     │    Clustering: silhouette ≥ 0.30│  │  │
│  │  │  each algorithm      │     │  • (Optional) testing_agent     │  │  │
│  │  └──────────────────────┘     │    runs on held-out 5% set      │  │  │
│  │              │                └──────────────────────────────────┘  │  │
│  │              └── → track_metrics → (loop)          │                │  │
│  │                                                     ▼                │  │
│  │                                                   END                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  SUPPORTED ALGORITHMS                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│  Classification:  RandomForest, GradientBoosting, LogisticRegression,      │
│                   SVC, XGBoost, ExtraTrees, AdaBoost, KNN,                 │
│                   DecisionTree, LightGBM                                    │
│                                                                             │
│  Regression:      RandomForest, GradientBoosting, Ridge, Lasso,            │
│                   XGBoost, ExtraTrees, SVR, AdaBoost, KNN, LightGBM        │
│                                                                             │
│  Clustering:      KMeans, AgglomerativeClustering, GaussianMixture,        │
│                   DBSCAN, IsolationForest                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Structure

Every run creates a **unique session folder** inside your output directory. All files for that run live together in one place.

```
ada_output/
└── abc12345/                              ← unique session folder (8-char ID)
    ├── best_XGBoostClassifier_f1_0.8934.pkl        ← best trained model
    ├── best_XGBoostClassifier_f1_0.8934_features.pkl  ← feature column list
    ├── summary.json                       ← complete run report
    ├── optimization_history.json          ← scores for every iteration
    ├── generated_code/                    ← all Python scripts Ada wrote
    │   ├── step1a_general.py              ← data understanding
    │   ├── step1b_profiling.py            ← task-specific profiling
    │   ├── step2_analysis.py              ← data cleaning + analysis
    │   ├── step3_ml.py                    ← baseline model (iteration 0)
    │   ├── step3_ml_iter1.py              ← first optimization pass
    │   └── step3_ml_iter2.py              ← second optimization pass (if run)
    └── data/
        └── your_data.csv                  ← working copy of your dataset
```

**The model filename tells you exactly what it is:**
`best_{Algorithm}_{metric}_{score}.pkl`
- `best_RandomForestClassifier_f1_0.8901.pkl` — Random Forest, F1=0.8901
- `best_XGBoostClassifier_f1_0.9234.pkl` — XGBoost, F1=0.9234
- `best_Ridge_r2_0.7654.pkl` — Ridge regression, R²=0.7654

### summary.json structure

```json
{
  "session_id": "abc12345",
  "task_type": "supervised_classification",
  "target_column": "churn",
  "dataset": "/path/to/data.csv",
  "status": "completed",
  "elapsed_seconds": 127.5,
  "best_model": {
    "algorithm": "XGBoostClassifier",
    "primary_metric": "f1",
    "score": 0.8934,
    "metrics": {
      "accuracy": 0.9012,
      "f1": 0.8934,
      "precision": 0.8865,
      "recall": 0.8980
    }
  },
  "evaluation": {
    "verdict": "pass",
    "strengths": ["high recall", "robust to noise"],
    "weaknesses": ["slight overfitting on training set"],
    "suggestions": ["try ensemble stacking for further gain"]
  },
  "optimization_history": [
    {"iteration": 0, "algorithm": "RandomForest", "primary_score": 0.72},
    {"iteration": 1, "algorithm": "XGBoostClassifier", "primary_score": 0.8934}
  ],
  "token_usage": {
    "data_understanding": {"prompt": 3200, "completion": 800},
    "ml_engineer": {"prompt": 8100, "completion": 2300},
    "testing_agent": {"prompt": 4500, "completion": 1200},
    "total_prompt": 50000,
    "total_completion": 12000
  },
  "output_dir": "/path/to/ada_output/abc12345",
  "model_path": "/path/to/ada_output/abc12345/best_XGBoostClassifier_f1_0.8934.pkl"
}
```

### Using the trained model

Always use `result["model_path"]` — the filename includes the algorithm and score so you never need to guess which file is the best model.

```python
import pickle
import pandas as pd

# Use the path returned by run_pipeline (includes algorithm + score in name)
model_path = result["model_path"]
# e.g. "ada_output/abc12345/best_XGBoostClassifier_f1_0.8934.pkl"

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the feature list (same name, _features suffix)
features_path = model_path.replace(".pkl", "_features.pkl")
with open(features_path, "rb") as f:
    features = pickle.load(f)

# Predict — align your data to the exact same columns used in training
new_data = pd.read_csv("new_data.csv")
for col in features:
    if col not in new_data.columns:
        new_data[col] = 0          # add missing columns with 0
X = new_data[features]             # keep only training columns, in order

predictions = model.predict(X)
probabilities = model.predict_proba(X)  # for classifiers only
```

---

## Configuration Reference

### .env file options

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Optional — Model
LLM_MODEL=gpt-4o-mini              # any OpenAI model ID

# Optional — Pipeline limits
MAX_RETRIES=10                     # max script execution retries per step
MAX_OPTIMIZATION_LOOPS=3           # distinct algorithms to try after baseline
MAX_TUNE_ITERATIONS=2              # hyperparameter tuning rounds per algorithm

# GitLab integration only
GITLAB_URL=https://gitlab.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxx
POLL_INTERVAL=60                   # seconds between issue polls
```

### Tuning the optimization budget

| Setting | Effect | Recommendation |
|---------|--------|----------------|
| `MAX_OPTIMIZATION_LOOPS=1` | Try 1 algorithm after baseline | Fast, cheap (dev/testing) |
| `MAX_OPTIMIZATION_LOOPS=3` | Try 3 algorithms | Default balance |
| `MAX_OPTIMIZATION_LOOPS=5` | Try 5 algorithms | Best model quality |
| `MAX_TUNE_ITERATIONS=1` | Tune once per algorithm | Faster |
| `MAX_TUNE_ITERATIONS=3` | Tune three times | More thorough |

### LLM model options

| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| `gpt-4o-mini` | Fast | Low | Good (default) |
| `gpt-4o` | Medium | Medium | Better |
| `gpt-4-turbo` | Slow | High | Best |

---

## Troubleshooting

### Common issues

**`ada: command not found`**
```bash
# Reinstall the package
cd ada-ds && pip install -e .
# Or run directly
python -m ada.cli run --dataset data.csv --instructions instructions.md
```

**`OPENAI_API_KEY not found`**
```bash
# Create .env in your working directory
echo "OPENAI_API_KEY=sk-proj-..." > .env
# Or pass it explicitly
ada run -d data.csv -i instructions.md -e /path/to/.env
```

**`ModuleNotFoundError: No module named 'xgboost'`**
```bash
pip install xgboost lightgbm catboost
```

**Pipeline fails at Step 2 (data cleaning)**
- Check your CSV has no completely empty columns
- Ensure the `Target Variable` in instructions matches an actual column name (case-sensitive)
- Try adding `## Data Notes` to describe known data issues

**Model file not found after run**
- All output files are inside the session subfolder: `ada_output/{session_id}/`
- Use `result["model_path"]` or `result["session_dir"]` from the Python API to get the exact path
- Check `summary.json` → `"model_path"` field for the full path

**GitLab: issues not picked up**
- Confirm the issue has exactly the label `ml-ready` (case-sensitive)
- Verify `GITLAB_TOKEN` has `api` scope
- Check `GITLAB_URL` matches your GitLab instance (including self-hosted URLs)
- Confirm the project is registered: `GET http://localhost:8001/projects`

**Web UI: blank screen / CORS errors**
- Ensure backend is running on port 8000: `curl http://localhost:8000/health`
- Frontend must be served from port 5173 (default Vite port)
- Check browser console for the actual error message

**Model quality too low (verdict: retry)**
- Increase `MAX_OPTIMIZATION_LOOPS` and `MAX_TUNE_ITERATIONS` in `.env`
- Add algorithm preferences in `## ML Requirements` (e.g., "try XGBoost", "maximize recall")
- At the Human Approval Gate, type feedback like "prioritize recall, try ensemble methods"
- Use a stronger LLM model: `LLM_MODEL=gpt-4o`

### Getting help

- All generated code is saved in `generated_code/` — inspect it to understand exactly what Ada did
- `summary.json` contains the full trace including `optimization_history` and per-agent `token_usage`
- `optimization_history.json` shows the score at every iteration so you can see how improvement happened
