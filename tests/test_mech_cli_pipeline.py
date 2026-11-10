import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


class MechCliPipelineTests(unittest.TestCase):
    def cli(self, *args, expected=0, module='robotprobe'):
        proc = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        self.assertNotIn('Traceback', proc.stderr)
        return json.loads(proc.stdout)

    def test_fixture_pipeline_preserves_provenance_and_emits_valid_run_traces(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            traces = []
            for label in ('clean', 'perturbed'):
                path = root / f'{label}.jsonl'
                with TraceRecorder(path, run_id=label, target='test-policy', scenario='instruction_swap',
                                   replay_level=ReplayLevel.TRACE_PLAYBACK) as recorder:
                    recorder.event(0, 'raw_action', 'policy_forward', value=[1, 0])
                    recorder.event(14, 'raw_action', 'policy_forward',
                                   value=[0, 1] if label == 'perturbed' else [1, 0])
                traces.append(path)
            config = root / 'capture.json'
            config.write_text(json.dumps({
                'format': 'robotprobe.mech.capture_config.v0', 'id': 'cli-regression',
                'model_access': 'white_box', 'layers': [{'name': 'layer'}],
                'storage': {'format': 'npy', 'dtype': 'float32', 'max_bytes_per_run': 4096},
                'claim_boundary': {'no_feature_labels': True, 'no_causal_claims': True},
            }), encoding='utf-8')
            dictionary = root / 'dictionary.json'
            dictionary.write_text(json.dumps({
                'format': 'robotprobe.mech.dictionary.v0', 'id': 'direction', 'type': 'manual_direction',
                'features': [{'feature_id': 'f0', 'label': 'test direction',
                              'layer': 'layer', 'direction': [1, 0, 0]}],
            }), encoding='utf-8')
            validation = self.cli('mech', 'validate', '--capture-config', config)
            self.assertTrue(validation['valid'])
            mech_dir = root / 'mech'
            collected = self.cli('mech', 'collect', '--target', 'synthetic', '--clean', traces[0],
                                 '--perturbed', traces[1], '--capture-config', config, '--out', mech_dir)
            self.assertTrue(collected['synthetic'])
            self.assertEqual(set(collected['activation_events']), {'clean', 'perturbed'})
            for label, path in collected['activation_events'].items():
                validation = self.cli('validate-trace', path)
                self.assertTrue(validation['valid'])
                self.assertEqual(validation['run_ids'], [label])
            cache = self.cli('mech', 'cache-info', '--mech-dir', mech_dir)
            self.assertGreater(cache['capture_count'], 0)
            summary = self.cli('summarize', '--clean', traces[0], '--perturbed', traces[1])
            summary_path = root / 'summary.json'
            summary_path.write_text(json.dumps(summary), encoding='utf-8')
            diff_path = root / 'diff.json'
            diff = self.cli('mech', 'diff-activations', '--mech-dir', mech_dir, '--out', diff_path)
            self.assertTrue(diff['synthetic'])
            feature_path = root / 'features.json'
            features = self.cli('mech', 'diff-features', '--mech-dir', mech_dir,
                                '--dictionary', dictionary, '--out', feature_path)
            self.assertTrue(features['synthetic'])
            self.assertFalse(features['evidence_eligible'])
            faith = self.cli('mech', 'faithfulness', '--mech-dir', mech_dir,
                             '--summary', summary_path, '--target-metric', 'raw_action_delta',
                             '--out', root / 'faithfulness.json')
            self.assertTrue(faith['synthetic'])
            self.assertFalse(faith['evidence_eligible'])
            self.assertEqual(faith['evidence_level'], 0)
            report_args = ['mech', 'report', '--summary', summary_path, '--activation-diff', diff_path,
                           '--feature-diff', feature_path]
            report_path = root / 'report.json'
            report = self.cli(*report_args, '--out', report_path)
            self.assertTrue(report['synthetic'])
            self.assertFalse(report['evidence_eligible'])
            self.assertEqual(report['evidence_level'], 0)
            self.cli(*report_args, '--out', root / 'report.md')
            markdown = (root / 'report.md').read_text(encoding='utf-8')
            self.assertIn('synthetic', markdown.lower())
            self.assertIn('test direction', markdown)
            checked = self.cli('check-mech', '--challenge',
                               'robotprobe-mech-v0-instruction-internal-leadtime', '--report', report_path,
                               expected=1, module='robotsec_range')
            self.assertFalse(checked['passed'])


if __name__ == '__main__':
    unittest.main()
