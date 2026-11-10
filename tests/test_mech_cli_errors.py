import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MechCliErrorTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, '-m', 'robotprobe', 'mech', *map(str, args)],
            capture_output=True, text=True, check=False,
        )

    def assert_structured_error(self, result, command):
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('Traceback', result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['format'], 'robotprobe.mech.error.v0')
        self.assertEqual(payload['command'], command)
        self.assertEqual(payload['status'], 'error')
        self.assertTrue(payload['error'])

    def test_missing_capture_config_is_a_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.run_cli('validate', '--capture-config', Path(td) / 'missing.yaml')
        self.assert_structured_error(result, 'validate')

    def test_invalid_activation_index_json_is_a_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'activation_index.json').write_text('{broken', encoding='utf-8')
            result = self.run_cli('diff-activations', '--mech-dir', root, '--out', root / 'diff.json')
        self.assert_structured_error(result, 'diff-activations')

    def test_missing_report_input_is_a_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.run_cli('report', '--summary', root / 'missing.json',
                                  '--activation-diff', root / 'diff.json', '--out', root / 'report.md')
        self.assert_structured_error(result, 'report')

    def test_malformed_yaml_is_a_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'broken.yaml'
            path.write_text('layers: [unterminated', encoding='utf-8')
            result = self.run_cli('validate', '--capture-config', path)
        self.assert_structured_error(result, 'validate')


if __name__ == '__main__':
    unittest.main()
