"""`robotprobe demo` — the first-run path, and the guards around its --out.

The refusal tests exist because of issue #40: emptying a caller-supplied output
directory is a destructive-action bug. They assert *both* that the command
refuses and that nothing was deleted, which is the part that actually matters.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEMO_ARTIFACTS = (
    "report.md",
    "bundle.json",
    "summary.json",
    "causal_diff.json",
    "check.json",
    "clean.validation.json",
    "perturbed.validation.json",
    "replay_l0.json",
    "bundle.command.json",
)


def run_demo(out) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "robotprobe", "demo", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )


def fake_project_root(base: Path) -> Path:
    """A directory that looks like someone else's tree, to aim --out at.

    Deliberately not the real repo root: if the guard were broken, pointing
    --out at the actual checkout would delete the workspace instead of failing
    a test.
    """
    base.mkdir(parents=True, exist_ok=True)
    (base / "pyproject.toml").write_text("[project]\nname = 'someone-elses'\n")
    (base / "src").mkdir()
    (base / "src" / "app.py").write_text("print('do not delete me')\n")
    return base


class DemoCommandTests(unittest.TestCase):
    def test_demo_writes_the_full_artifact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            proc = run_demo(out)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            for name in DEMO_ARTIFACTS:
                self.assertTrue((out / name).exists(), f"missing artifact: {name}")
            self.assertIn("Gripper Freeze", (out / "report.md").read_text(encoding="utf-8"))

    def test_demo_prints_where_it_wrote(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_demo(Path(tmp) / "demo")
            self.assertIn("RobotProbe quickstart demo wrote", proc.stdout)
            self.assertIn("report.md", proc.stdout)

    def test_demo_is_repeatable_on_its_own_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            self.assertEqual(run_demo(out).returncode, 0)
            second = run_demo(out)
            self.assertEqual(second.returncode, 0, "demo must not refuse its own marker")
            self.assertTrue((out / "report.md").exists())


class DemoOutGuardTests(unittest.TestCase):
    def test_refuses_a_directory_with_foreign_entries(self):
        """Regression for issue #40."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "not-ours"
            out.mkdir()
            keep = out / "important.txt"
            keep.write_text("do not delete")
            (out / "sub").mkdir()
            (out / "sub" / "x.txt").write_text("keep")

            proc = run_demo(out)

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing", (proc.stdout + proc.stderr).lower())
            self.assertTrue(keep.exists(), "foreign file was deleted")
            self.assertTrue((out / "sub" / "x.txt").exists(), "foreign subdirectory was deleted")

    def test_refuses_a_directory_that_looks_like_a_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_project_root(Path(tmp) / "someone-elses-tree")
            proc = run_demo(root)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue((root / "pyproject.toml").exists())
            self.assertTrue((root / "src" / "app.py").exists())

    def test_refuses_when_out_is_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "afile"
            target.write_text("content")
            proc = run_demo(target)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(target.read_text(), "content")


class DemoAssetTests(unittest.TestCase):
    def test_packaged_demo_assets_match_the_source_copies(self):
        """The demo ships its inputs as package data; pin the copies together."""
        pairs = {
            ROOT / "robotprobe/data/demo/clean.jsonl": ROOT / "examples/quickstart/traces/clean.jsonl",
            ROOT / "robotprobe/data/demo/gripper_freeze_mid.jsonl":
                ROOT / "examples/quickstart/traces/gripper_freeze_mid.jsonl",
            ROOT / "robotprobe/data/demo/intervention.yaml": ROOT / "interventions/gripper_freeze_mid.yaml",
        }
        for packaged, source in pairs.items():
            self.assertTrue(packaged.exists(), f"packaged demo asset missing: {packaged}")
            self.assertEqual(packaged.read_bytes(), source.read_bytes(),
                             f"packaged asset drifted from source: {packaged.name}")


class GuardedCommandTests(unittest.TestCase):
    """The CLI must not advertise capabilities it does not implement."""

    def _run(self, *argv) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "robotprobe", *argv],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_mech_help_lists_only_implemented_commands(self):
        proc = self._run("mech", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("NOT IMPLEMENTED", proc.stdout)
        self.assertNotIn("train-sae", proc.stdout)
        self.assertNotIn("steer", proc.stdout)


if __name__ == "__main__":
    unittest.main()
