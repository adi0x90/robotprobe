import json
import os
import signal
import sys
import time
import tempfile
import unittest
from pathlib import Path

from robotprobe.integrations.vla_eval.model_server_smoke import (
    VlaEvalServeSpec,
    build_serve_command,
    start_server_process,
    terminate_process,
    write_attempt_result,
)


class VlaEvalModelServerSmokeTests(unittest.TestCase):
    def test_build_serve_command_uses_official_vla_eval_serve_config(self):
        spec = VlaEvalServeSpec(
            config_path=Path("/opt/vla-evaluation-harness/configs/model_servers/openvla/libero_spatial.yaml"),
            cwd=Path("/opt/vla-evaluation-harness"),
            port=8123,
            host="127.0.0.1",
            vla_eval_executable="/venv/bin/vla-eval",
            arg_overrides=("model_path=openvla/test-checkpoint",),
        )

        command = build_serve_command(spec)

        self.assertEqual(command[:3], ["/venv/bin/vla-eval", "serve", "--config"])
        self.assertIn("/opt/vla-evaluation-harness/configs/model_servers/openvla/libero_spatial.yaml", command)
        self.assertIn("--arg", command)
        self.assertIn("port=8123", command)
        self.assertIn("host=127.0.0.1", command)
        self.assertIn("model_path=openvla/test-checkpoint", command)

    def test_write_attempt_result_keeps_model_server_log_tail(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            log_path = out_dir / "server.log"
            log_path.write_text("\n".join(f"line-{i}" for i in range(20)) + "\n", encoding="utf-8")

            result_path = write_attempt_result(
                out_dir=out_dir,
                status="failed",
                model_name="openvla-libero-spatial",
                server_config=Path("/configs/openvla.yaml"),
                vla_eval_cwd=Path("/repo/vla-evaluation-harness"),
                command=["vla-eval", "serve", "--config", "/configs/openvla.yaml"],
                log_path=log_path,
                stage="startup",
                error="server exited before opening websocket",
                log_tail_lines=3,
            )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["model_server"]["name"], "openvla-libero-spatial")
            self.assertEqual(result["failure"]["stage"], "startup")
            self.assertEqual(result["server_log_tail"], ["line-17", "line-18", "line-19"])

    def test_terminate_process_stops_nested_uv_run_child(self):
        self._assert_nested_child_stopped(ignore_sigterm=False, parent_exits=False)

    def test_terminate_process_kills_resistant_child_after_parent_exits(self):
        self._assert_nested_child_stopped(ignore_sigterm=True, parent_exits=False)

    def test_terminate_process_cleans_group_after_leader_already_exited(self):
        self._assert_nested_child_stopped(ignore_sigterm=True, parent_exits=True)

    def _assert_nested_child_stopped(self, *, ignore_sigterm, parent_exits):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            child_pid_path = out_dir / "child.pid"
            log_path = out_dir / "server.log"
            child_script = (
                "import os, pathlib, signal, time\n"
                + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_sigterm else "")
                + f"path = pathlib.Path({str(child_pid_path)!r})\n"
                "pending = path.with_suffix('.pending')\n"
                "pending.write_text(str(os.getpid()), encoding='utf-8')\n"
                "pending.replace(path)\n"
                "time.sleep(60)\n"
            )
            script = (
                "import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
                + ("" if parent_exits else "time.sleep(60)\n")
            )
            proc = start_server_process(
                [sys.executable, "-c", script],
                cwd=out_dir,
                log_path=log_path,
                env=os.environ.copy(),
            )
            child_pid = None
            try:
                child_pid = _read_pid(child_pid_path)
                if parent_exits:
                    proc.wait(timeout=5)
                terminate_process(proc, timeout=0.2)

                self.assertIsNotNone(proc.poll())
                # The group kill is asynchronous: give the OS a moment to reap the grandchild.
                deadline = time.monotonic() + 5
                while _pid_is_alive(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_pid_is_alive(child_pid), f"child process survived: {child_pid}")
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
                if child_pid is not None:
                    _kill_if_alive(child_pid)


if __name__ == "__main__":
    unittest.main()


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return int(path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"pid file was not written: {path}")


def _pid_is_alive(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
        except OSError:
            fields = []
        if len(fields) > 2 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_if_alive(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
