"""GitLab REST API client — file download, issue management, uploads."""

import re
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Labels used across the issue lifecycle
LABEL_READY      = "ml-ready"
LABEL_PROCESSING = "ml-processing"
LABEL_COMPLETE   = "ml-complete"
LABEL_FAILED     = "ml-failed"


class GitLabClient:
    def __init__(self, base_url: str, token: str, project_id: int, project_path: str = ""):
        self.base_url     = base_url.rstrip("/")
        self.project_id   = project_id
        self.project_path = project_path.strip("/")
        self._token       = token
        self._headers     = {"PRIVATE-TOKEN": token}

    # ── Issues ────────────────────────────────────────────────────────────

    def get_open_issues(self, label: str = LABEL_READY) -> list[dict]:
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/issues"
        resp = httpx.get(
            url,
            headers=self._headers,
            params={"labels": label, "state": "opened", "per_page": 50},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def post_comment(self, issue_iid: int, body: str) -> dict:
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/issues/{issue_iid}/notes"
        resp = httpx.post(url, headers=self._headers, json={"body": body}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def update_issue(
        self,
        issue_iid: int,
        add_labels: str = "",
        remove_labels: str = "",
        close: bool = False,
    ):
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/issues/{issue_iid}"
        data: dict = {}
        if add_labels:
            data["add_labels"] = add_labels
        if remove_labels:
            data["remove_labels"] = remove_labels
        if close:
            data["state_event"] = "close"
        resp = httpx.put(url, headers=self._headers, json=data, timeout=30)
        resp.raise_for_status()

    def claim_issue(self, issue_iid: int):
        """Swap ml-ready → ml-processing atomically."""
        self.update_issue(
            issue_iid,
            add_labels=LABEL_PROCESSING,
            remove_labels=LABEL_READY,
        )

    def complete_issue(self, issue_iid: int):
        self.update_issue(
            issue_iid,
            add_labels=LABEL_COMPLETE,
            remove_labels=LABEL_PROCESSING,
            close=True,
        )

    def fail_issue(self, issue_iid: int):
        self.update_issue(
            issue_iid,
            add_labels=LABEL_FAILED,
            remove_labels=LABEL_PROCESSING,
        )

    # ── File transfers ────────────────────────────────────────────────────

    def download_attachment(self, attachment_url: str, dest: Path):
        """Download a GitLab upload path (relative or absolute) to *dest*.

        /uploads/ paths on GitLab.com are web-app routes that redirect to the
        sign-in page when hit directly. Route them through the projects API
        (/api/v4/projects/{id}/uploads/...) instead — it accepts PRIVATE-TOKEN.
        """
        if attachment_url.startswith("http"):
            full_url = attachment_url
        elif attachment_url.startswith("/uploads/"):
            # attachment_url = "/uploads/{secret}/{file}" — strip the leading
            # "/uploads" so we don't duplicate it in the API path.
            suffix = attachment_url[len("/uploads"):]
            full_url = f"{self.base_url}/api/v4/projects/{self.project_id}/uploads{suffix}"
        else:
            full_url = f"{self.base_url}{attachment_url}"
        logger.info("Downloading attachment: %s", full_url)
        resp = httpx.get(
            full_url,
            headers=self._headers,
            follow_redirects=True,
            timeout=120,
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("Saved %d bytes → %s", len(resp.content), dest)

    def upload_file(self, file_path: Path) -> dict:
        """Upload a file to project uploads. Returns GitLab upload dict {alt, url, markdown}."""
        url = f"{self.base_url}/api/v4/projects/{self.project_id}/uploads"
        with open(file_path, "rb") as fh:
            resp = httpx.post(
                url,
                headers=self._headers,
                files={"file": (file_path.name, fh)},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()

    # ── Issue body parsing ────────────────────────────────────────────────

    @staticmethod
    def extract_attachment(body: str, extension: str) -> tuple[str, str] | None:
        """Return (filename, relative_url) for the first attachment with the given extension."""
        # GitLab embeds attachments as: [name.ext](/uploads/hash/name.ext)
        pattern = rf'\[([^\]]+\.{re.escape(extension)})\]\((/uploads/[^\)]+\.{re.escape(extension)})\)'
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
        return None

    # ── Project lookup ────────────────────────────────────────────────────

    def lookup_project_id(self, namespace_with_path: str) -> int:
        """Resolve 'group/repo' → numeric project ID."""
        import urllib.parse
        encoded = urllib.parse.quote(namespace_with_path, safe="")
        url = f"{self.base_url}/api/v4/projects/{encoded}"
        resp = httpx.get(url, headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["id"]
