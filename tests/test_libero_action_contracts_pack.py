import json
import unittest
from pathlib import Path

from robotprobe.packs import build_claim_card, test_pack, validate_pack


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "libero-action-contracts"


class LiberoActionContractsPackTests(unittest.TestCase):
    def test_libero_action_contracts_pack_validates_and_has_golden_reports(self):
        report = validate_pack(PACK)
        pack_test = test_pack(PACK)

        self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))
        self.assertTrue(pack_test.valid, json.dumps(pack_test.to_dict(), indent=2))
        self.assertTrue((PACK / "golden" / "summaries" / "gripper_freeze_mid.json").exists())
        self.assertTrue((PACK / "golden" / "reports" / "gripper_freeze_mid.md").exists())

    def test_libero_pack_claim_card_keeps_model_failure_non_claim_explicit(self):
        card = build_claim_card(PACK)

        self.assertIn("LIBERO action-contract", card)
        self.assertIn("action-postprocess", card)
        self.assertIn("does not demonstrate", card)
        self.assertIn("model failure", card)
