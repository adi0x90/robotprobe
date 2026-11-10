#!/usr/bin/env python3
"""Prepare a local tokenizer from Google's public, OpenPI-supported distribution.

Build in a separate tokenizer-only environment (no model/GPU dependency):
    uv run --no-project --with transformers==4.53.2 --with sentencepiece \
      --with protobuf --with numpy scripts/prepare_pi05_public_tokenizer.py \
      prepare --out /tmp/pi05-tokenizer

Verify again in the actual LeRobot environment before inference:
    python scripts/prepare_pi05_public_tokenizer.py verify \
      --tokenizer /tmp/pi05-tokenizer --verify-lerobot

The SentencePiece file matches the pinned HF artifact through its public Git
LFS pointer identity. The locally generated HF wrapper is a separate artifact;
this script does not claim to have fetched the gated HF wrapper configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


PUBLIC_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
MODEL_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
MODEL_SIZE = 4_264_023
HF_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
HF_POINTER_OID = "22ad048d392b4f5191be7cec47d03eb3cdb360f4"
HF_METADATA_URL = (
    "https://huggingface.co/api/models/google/paligemma-3b-pt-224/tree/"
    f"{HF_REVISION}?recursive=true&expand=false"
)
TOKENIZER_REVISION = f"public-gcs-sha256-{MODEL_SHA256}"
BUILD_TRANSFORMERS_VERSION = "4.53.2"


def lfs_pointer_oid(sha256: str, size: int) -> str:
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n"
    ).encode()
    return hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest()


def verify_model(data: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256 or len(data) != MODEL_SIZE:
        raise ValueError("public tokenizer bytes do not match the pinned SHA256 and size")
    pointer_oid = lfs_pointer_oid(digest, len(data))
    if pointer_oid != HF_POINTER_OID:
        raise ValueError("public tokenizer does not match the pinned HF LFS pointer identity")
    return {"sha256": digest, "bytes": len(data), "git_lfs_pointer_oid": pointer_oid}


def probe_prompts() -> list[str]:
    tasks = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "put the red cup in the drawer",
        "open the top drawer of the cabinet",
        "  put_the_cup\nin the drawer  ",
        "move the café cup beside the 猫 bowl 🤖",
        "grasp\tthe handle",
    ]
    states = ["0 128 255 13 100 99 127 130", "-1 0 1 254 255 256 127 128", "128 " * 31 + "127"]
    prompts = [
        f"Task: {task.strip().replace('_', ' ').replace(chr(10), ' ')}, State: {state};\nAction: "
        for task in tasks
        for state in states
    ]
    # Also exercise empty input, whitespace and truncation, outside the normal task corpus.
    return prompts + ["", " ", "\n", "Task: " + "move the cup " * 256 + ";\nAction: "]


def verify_tokenizer(path: Path, *, verify_lerobot: bool = False) -> dict[str, Any]:
    import sentencepiece
    import transformers
    from transformers import AutoTokenizer

    identity = verify_model((path / "tokenizer.model").read_bytes())
    reference = sentencepiece.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    prompts = probe_prompts()
    expected = [reference.encode(prompt, add_bos=True, add_eos=False) for prompt in prompts]
    for prompt, token_ids in zip(prompts, expected, strict=True):
        if tokenizer.encode(prompt, add_special_tokens=True) != token_ids:
            raise ValueError("local tokenizer token IDs differ from the official SentencePiece reference")
    max_length = 200
    encoded = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="np",
    )
    padded_ids = [ids[:max_length] + [0] * max(0, max_length - len(ids)) for ids in expected]
    padded_mask = [
        [1] * min(len(ids), max_length) + [0] * max(0, max_length - len(ids)) for ids in expected
    ]
    if encoded["input_ids"].tolist() != padded_ids or encoded["attention_mask"].tolist() != padded_mask:
        raise ValueError("local tokenizer padding/truncation differs from the OpenPI reference")
    lerobot_result: dict[str, Any] = {"executed": False}
    if verify_lerobot:
        import importlib.metadata

        import torch
        from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
        from lerobot.types import TransitionKey
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        step = TokenizerProcessorStep(
            tokenizer=tokenizer,
            max_length=max_length,
            padding_side="right",
            padding="max_length",
        )
        transition = {
            TransitionKey.OBSERVATION: {"observation.state": torch.zeros((len(prompts), 8))},
            TransitionKey.COMPLEMENTARY_DATA: {"task": prompts},
        }
        output = step(transition)[TransitionKey.OBSERVATION]
        if output[OBS_LANGUAGE_TOKENS].tolist() != padded_ids:
            raise ValueError("actual LeRobot processor token IDs differ from the SentencePiece reference")
        if output[OBS_LANGUAGE_ATTENTION_MASK].int().tolist() != padded_mask:
            raise ValueError("actual LeRobot processor attention mask differs from the reference")
        lerobot_result = {
            "executed": True,
            "version": importlib.metadata.version("lerobot"),
            "processor": "TokenizerProcessorStep",
            "passed": True,
        }
    return {
        "format": "robotprobe.pi05.public_tokenizer_verification.v0",
        "passed": True,
        "source": PUBLIC_URL,
        "model": identity,
        "tokenizer_revision": TOKENIZER_REVISION,
        "transformers_version": transformers.__version__,
        "sentencepiece_version": sentencepiece.__version__,
        "prompt_cases": len(prompts),
        "tokens_equal_to_openpi_sentencepiece": True,
        "padding_and_truncation_equal": True,
        "lerobot": lerobot_result,
        "boundary": "SentencePiece identity and tested tokenization, not the gated HF wrapper configuration",
    }


def prepare_tokenizer(out: Path) -> dict[str, Any]:
    import transformers
    from transformers import GemmaTokenizerFast

    if transformers.__version__ != BUILD_TRANSFORMERS_VERSION:
        raise ValueError(
            f"build with transformers=={BUILD_TRANSFORMERS_VERSION} in a separate tokenizer-only environment; "
            "the exported tokenizer can then be verified in the actual runtime environment"
        )
    if out.exists() and any(out.iterdir()):
        raise ValueError("tokenizer output directory must be absent or empty")
    with urlopen(PUBLIC_URL, timeout=60) as response:
        data = response.read(MODEL_SIZE + 1)
    identity = verify_model(data)
    with urlopen(HF_METADATA_URL, timeout=30) as response:
        entries = json.load(response)
    entry = next(item for item in entries if item.get("path") == "tokenizer.model")
    if entry.get("oid") != HF_POINTER_OID or entry.get("size") != MODEL_SIZE:
        raise ValueError("public HF metadata does not match the pinned tokenizer identity")
    out.mkdir(parents=True, exist_ok=True)
    model = out / "tokenizer.model"
    model.write_bytes(data)
    tokenizer = GemmaTokenizerFast(vocab_file=str(model), add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(out)
    verification = verify_tokenizer(out)
    receipt = {
        "format": "robotprobe.pi05.public_tokenizer_preparation.v0",
        "source": PUBLIC_URL,
        "source_identity": identity,
        "hf_metadata_url": HF_METADATA_URL,
        "hf_sentencepiece_revision": HF_REVISION,
        "hf_sentencepiece_pointer_match": True,
        "wrapper_source": "locally built transformers.GemmaTokenizerFast",
        "wrapper_build_version": transformers.__version__,
        "wrapper_options": {"add_bos_token": True, "add_eos_token": False, "padding_side": "right"},
        "files": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(out.iterdir())
            if file.is_file()
        },
        "verification": verification,
        "boundary": "HF tokenizer.model byte identity is verified; the HF wrapper config was not downloaded",
    }
    (out / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--tokenizer", type=Path, required=True)
    verify.add_argument("--verify-lerobot", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            prepare_tokenizer(args.out)
            if args.command == "prepare"
            else verify_tokenizer(args.tokenizer, verify_lerobot=args.verify_lerobot)
        )
    except (ValueError, OSError, ImportError) as error:
        print(json.dumps({"passed": False, "error": str(error), "type": type(error).__name__}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
