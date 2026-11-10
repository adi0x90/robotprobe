import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pi05_public_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("prepare_pi05_public_tokenizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tokenizer_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tokenizer_script)


class Pi05PublicTokenizerTests(unittest.TestCase):
    def test_pinned_public_model_reconstructs_the_pinned_hf_git_lfs_pointer(self):
        self.assertEqual(
            tokenizer_script.lfs_pointer_oid(tokenizer_script.MODEL_SHA256, tokenizer_script.MODEL_SIZE),
            tokenizer_script.HF_POINTER_OID,
        )

    def test_untrusted_or_truncated_model_is_rejected_before_conversion(self):
        for data in (b"", b"not a tokenizer", b"x" * tokenizer_script.MODEL_SIZE):
            with self.subTest(size=len(data)), self.assertRaisesRegex(ValueError, "pinned SHA256"):
                tokenizer_script.verify_model(data)
