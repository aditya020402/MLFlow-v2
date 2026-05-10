"""Parse a .md instructions file into structured fields for the ML pipeline."""

import re


TASK_TYPE_MAP = {
    "classification":           "supervised_classification",
    "supervised_classification": "supervised_classification",
    "regression":               "supervised_regression",
    "supervised_regression":    "supervised_regression",
    "clustering":               "unsupervised",
    "unsupervised":             "unsupervised",
}


def parse(body: str) -> dict:
    """
    Recognised headings (case-insensitive):
        ## Task Type
        ## Target Variable
        ## Business Context
        ## Data Notes
        ## ML Requirements
        ## Evaluation Criteria
        ## Validation

    Returns a dict with keys:
        task_type, target_column, validate, human_feedback,
        dataset_description, data_notes, ml_requirements,
        evaluation_criteria (dict), raw (original body)
    """
    result = {
        "raw":                  body,
        "task_type":            None,
        "target_column":        None,
        "validate":             False,
        "dataset_description":  "",
        "data_notes":           [],
        "ml_requirements":      [],
        "evaluation_criteria":  {},
        "human_feedback":       "",
    }

    sections = re.split(r"(?m)^##\s+", body)

    for section in sections:
        if not section.strip():
            continue
        lines   = section.strip().splitlines()
        heading = lines[0].strip().lower()
        content = "\n".join(lines[1:]).strip()

        if "task type" in heading:
            raw = content.split("\n")[0].strip().lower()
            result["task_type"] = TASK_TYPE_MAP.get(raw)

        elif "target variable" in heading or "target column" in heading:
            m = re.search(r"`([^`]+)`", content)
            if m:
                result["target_column"] = m.group(1).strip()
            else:
                m2 = re.search(r"(?:column|target|variable)\s*[:\-]\s*(\w+)", content, re.I)
                result["target_column"] = m2.group(1).strip() if m2 else content.split()[0]

        elif "business context" in heading or "context" in heading:
            result["dataset_description"] = content

        elif "data notes" in heading or "notes" in heading:
            items = re.findall(r"^[-*]\s+(.+)$", content, re.MULTILINE)
            result["data_notes"] = items or [content] if content else []

        elif "ml requirements" in heading or "requirements" in heading:
            items = re.findall(r"^[-*]\s+(.+)$", content, re.MULTILINE)
            result["ml_requirements"] = items or [content] if content else []

        elif "validation" in heading:
            raw_val = content.split("\n")[0].strip().lower()
            result["validate"] = raw_val in {"yes", "true", "enabled", "on", "1"}

        elif "evaluation" in heading:
            m = re.search(r"primary\s+metric\s*[:\-]\s*(\w+)", content, re.I)
            if m:
                result["evaluation_criteria"]["primary_metric"] = m.group(1)
            m2 = re.search(r"threshold\s*[:\-]\s*([\d.]+)", content, re.I)
            if m2:
                result["evaluation_criteria"]["threshold"] = float(m2.group(1))

    parts: list[str] = []
    if result["dataset_description"]:
        parts.append(f"Business Context: {result['dataset_description']}")
    if result["ml_requirements"]:
        parts.append("ML Requirements:\n" + "\n".join(f"- {r}" for r in result["ml_requirements"]))
    if result["data_notes"]:
        parts.append("Data Notes:\n" + "\n".join(f"- {n}" for n in result["data_notes"]))
    if result["evaluation_criteria"]:
        parts.append(f"Evaluation Criteria: {result['evaluation_criteria']}")
    result["human_feedback"] = "\n\n".join(parts)

    return result


def validation_error(parsed: dict) -> str | None:
    if not parsed.get("task_type"):
        return (
            "Could not determine Task Type. "
            "Please add a '## Task Type' section with one of: "
            "classification, regression, clustering."
        )
    return None
