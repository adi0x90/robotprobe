# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub Security Advisories:

<https://github.com/adi0x90/robotprobe/security/advisories/new>

If advisories are not available to you, open a minimal issue asking for a private channel and the
maintainer ([@adi0x90](https://github.com/adi0x90)) will follow up. Do not include exploit details
in a public issue.

Include what you were running, the version or commit, and the smallest input that reproduces the
problem. Traces, pack manifests, or capture configs that trigger the issue are welcome; remove any
data you are not able to share.

## Supported versions

The project is in alpha. Security fixes land on `main` and are noted in `CHANGELOG.md`; there are no
maintained release branches yet.

## What is in scope

RobotProbe parses untrusted input and can import third-party Python. Both are worth reporting:

- **Pack and plugin loading.** Packs declare Python entrypoints. `robotprobe pack test` refuses to
  import them unless `--allow-code` is passed; `robotprobe pack validate` must not execute pack code.
  A path that executes pack code without that flag is a vulnerability.
- **Trace and config parsing.** JSONL traces, intervention manifests, capture configs, pack
  manifests, and MCAP files are treated as untrusted input. Crashes are ordinary bugs; code
  execution, path traversal outside the output directory, or resource exhaustion are vulnerabilities.
- **Output directory handling.** Commands that write bundles must not delete or overwrite paths they
  did not create.
- **Exported artifacts.** A sidecar or MCAP export must not include files from outside the bundle.

## What is not in scope

- The bundled demo, fixtures, and example packs are deliberately synthetic.
- Faults that require the operator to pass `--allow-code`, or to point the tool at a pack they wrote
  themselves, are the documented behaviour of a code-execution feature, not a vulnerability.
- Findings that require modifying the local checkout before running it.

## Safety boundary

This project ships simulation-first intervention manifests. No hardware-destructive intervention is
included, and manifests declare `release_classification: sim_only` with `hardware_allowed: false`.
Running RobotProbe against real hardware is out of scope for the project's own safety story and is
the operator's responsibility.
