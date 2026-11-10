"""Repository hygiene checks that guard against accidentally re-publishing private data.

These are cheap text scans, not correctness tests. They exist because the project was
developed in a private checkout that contained maintainer notes, cloud endpoints, and
absolute paths, and those have a habit of reappearing in commit messages, fixtures, and
example docs.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".mcap", ".safetensors", ".npy", ".lock"}

# Local notes that are deliberately kept out of the published tree.
PRIVATE_NOTES_DIR = ROOT / "docs" / "pvt"
PRIVATE_NOTES_IGNORE_RULE = "/docs/pvt/"

# Each pattern is something that must never appear in a published file.
FORBIDDEN = [
    (re.compile(r"root@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"), "SSH endpoint with root login"),
    (re.compile(r"/Users/[A-Za-z]"), "absolute home directory path"),
    (re.compile(r"/root/[A-Za-z]"), "absolute root directory path"),
    (re.compile(r"/workspace/[A-Za-z]"), "absolute cloud workspace path"),
    (re.compile(r"docs/internal"), "reference to the private maintainer notes"),
    (re.compile(r"vla-redscore|vlrs-pi05"), "internal project codename"),
    (re.compile(r"adi0x90@gmail\.com"), "personal email address; use the GitHub contact instead"),
]


def _candidate_paths():
    """Files that would be published: tracked plus untracked-but-not-ignored.

    Gitignored local material (for example a private research directory kept beside the checkout)
    is never published, so scanning it would only make the check fail on the maintainer's machine.
    Outside a git checkout every file is a candidate.
    """
    if (ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        return [ROOT / name for name in proc.stdout.decode("utf-8").split("\0") if name]
    return list(ROOT.rglob("*"))


def _iter_text_files():
    for path in _candidate_paths():
        if path == SELF or not path.is_file():
            continue
        if PRIVATE_NOTES_DIR in path.parents:
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield path, text


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_private_endpoints_or_paths_in_the_tree(self):
        findings = []
        for path, text in _iter_text_files():
            for lineno, line in enumerate(text.splitlines(), start=1):
                for pattern, reason in FORBIDDEN:
                    if pattern.search(line):
                        findings.append(f"{path.relative_to(ROOT)}:{lineno}: {reason}: {line.strip()[:120]}")
        self.assertEqual(findings, [], "private data found:\n" + "\n".join(findings))

    def test_docs_directory_has_no_maintainer_notes(self):
        docs = ROOT / "docs"
        self.assertFalse((docs / "internal").exists(), "docs/internal must not be published")
        self.assertFalse((docs / "private-release.md").exists(), "private-release notes must not be published")

    def test_no_raw_cloud_run_artifacts_are_published(self):
        self.assertFalse(
            (ROOT / "docs" / "artifacts").exists(),
            "raw run logs belong in the private repository, not the published one",
        )

    def test_local_notes_are_ignored_and_never_tracked(self):
        """docs/pvt holds notes that must never be published.

        Two checks, because either alone is insufficient: the .gitignore entry is what makes the
        rule travel with the repository to every clone, and the ls-files check catches the case
        where someone deletes the rule and force-adds the directory anyway.
        """
        self.assertFalse(
            (ROOT / "docs" / "incidents").exists(),
            "case studies live in docs/pvt; docs/incidents must not reappear",
        )
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(PRIVATE_NOTES_IGNORE_RULE, gitignore, "docs/pvt must stay in .gitignore")

        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout")

        proc = subprocess.run(
            ["git", "ls-files", "docs/pvt"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            proc.stdout.strip(),
            "",
            "docs/pvt is tracked; it must stay out of the published tree",
        )
