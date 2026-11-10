from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

from robotprobe.integrations.isaac_lab_arena.trace_ingest import ingest_isaac_lab_arena_trace
from robotprobe.integrations.incident_datasets.robofac import convert_robofac_episodes, download_robofac_seed
from robotprobe.integrations.incident_datasets.vifailback import convert_vifailback_annotations, download_vifailback_seed
from robotprobe.integrations.ros2.mcap_adapter import ingest_ros2_mcap
from robotprobe.integrations.vla_eval.episode_ingest import ingest_vla_eval_episode
from robotprobe.bundle import create_incident_bundle
from robotprobe.contracts import check_contract_file
from robotprobe.doctor import run_doctor
from robotprobe.exporters import export_mcap_bundle, export_sidecar, import_mcap_bundle
from robotprobe.mech.diff import diff_activations
from robotprobe.mech.collect import collect_activations
from robotprobe.mech.features import diff_features
from robotprobe.mech.cache_view import build_cache_summary
from robotprobe.mech.faithfulness import compute_consequence_faithfulness
from robotprobe.mech.report import write_mech_report
from robotprobe.mech.schemas import load_capture_config, validate_capture_config
from robotprobe.packs import (
    PACK_TEMPLATES,
    build_claim_card,
    build_pack_registry,
    init_pack,
    list_pack,
    load_pack_registry,
    test_pack,
    validate_pack,
)
from robotprobe.replay import build_replay_plan
from robotprobe.summarize import summarize_trace_pair
from robotprobe.report import write_incident_report
from robotprobe.conformance import CONFORMANCE_LEVELS, check_conformance_file
from robotprobe.campaign import run_campaign_file
from robotprobe.worldmodel import check_world_model_faithfulness
from robotprobe.schema import validate_trace_file
from robotprobe.runtime.config import inspect_runtime_config, run_runtime_config
from robotprobe.runtime.evidence import create_runtime_evidence_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotprobe")
    sub = parser.add_subparsers(dest="command", required=True)

    # Registered first so it is the first thing a newcomer sees in --help.
    demo = sub.add_parser("demo", help="Run the bundled quickstart incident and write a report. Start here.")
    demo.add_argument("--out", default="robotprobe-demo-out", help="Output directory (default: ./robotprobe-demo-out)")

    summarize = sub.add_parser("summarize", help="Summarize a clean/perturbed trace pair.")
    summarize.add_argument("--clean", required=True)
    summarize.add_argument("--perturbed", required=True)
    summarize.add_argument("--manifest", default=None)
    summarize.add_argument("--metrics", default=None)

    report = sub.add_parser("report", help="Render a RobotProbe incident report.")
    report.add_argument("--summary", required=True)
    report.add_argument("--check", default=None)
    report.add_argument("--title", default="RobotProbe Incident Report")
    report.add_argument("--out", default=None)

    validate = sub.add_parser("validate-trace", help="Validate a RobotProbe JSONL trace.")
    validate.add_argument("trace")

    conformance = sub.add_parser("conformance", help="Check a trace against the failure-semantics conformance profile.")
    conformance.add_argument("trace")
    conformance.add_argument("--level", default=None, choices=list(CONFORMANCE_LEVELS), help="Required conformance level; exit non-zero unless met.")

    campaign = sub.add_parser("campaign", help="Run a fault-injection campaign against a baseline trace.")
    campaign_sub = campaign.add_subparsers(dest="campaign_command", required=True)
    campaign_run = campaign_sub.add_parser("run", help="Run a campaign spec and write a campaign bundle.")
    campaign_run.add_argument("--campaign", required=True)
    campaign_run.add_argument("--out", required=True)
    campaign_run.add_argument("--fail-on-violation", action="store_true", help="Exit non-zero if any fault violates a contract.")

    worldmodel = sub.add_parser("worldmodel", help="World-model faithfulness diagnostics for imagined evidence.")
    worldmodel_sub = worldmodel.add_subparsers(dest="worldmodel_command", required=True)
    wm_faith = worldmodel_sub.add_parser("faithfulness", help="Diff imagined world_model_output vs realized observation/outcome.")
    wm_faith.add_argument("--trace", required=True)
    wm_faith.add_argument("--state-tolerance", type=float, default=1e-3)
    wm_faith.add_argument("--divergence-threshold", type=float, default=0.5)
    wm_faith.add_argument("--out", default=None)
    wm_faith.add_argument("--fail-on-divergence", action="store_true", help="Exit non-zero if any imagined event diverged.")

    replay = sub.add_parser("replay", help="Build a replay plan for a trace at L0/L1/L2/L3.")
    replay.add_argument("trace")
    replay.add_argument("--level", required=True)
    replay.add_argument("--action-envelope", default=None)

    bundle = sub.add_parser("bundle", help="Write a complete RobotProbe incident bundle.")
    bundle.add_argument("--clean", required=True)
    bundle.add_argument("--perturbed", required=True)
    bundle.add_argument("--manifest", default=None)
    bundle.add_argument("--challenge", default=None)
    bundle.add_argument("--metrics", default=None)
    bundle.add_argument("--title", default="RobotProbe Incident Report")
    bundle.add_argument("--out", required=True)

    runtime = sub.add_parser("runtime", help="Inspect or run modular robot runtimes.")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_inspect = runtime_sub.add_parser(
        "inspect",
        help="Statically inspect runtime components and compatibility.",
    )
    runtime_inspect.add_argument("--config", required=True)
    runtime_run = runtime_sub.add_parser(
        "run",
        help="Run one configured modular robot episode.",
    )
    runtime_run.add_argument("--config", required=True)
    runtime_evidence = runtime_sub.add_parser(
        "evidence",
        help="Create a runtime-native evidence bundle from a trace.",
    )
    runtime_evidence.add_argument("--trace", required=True)
    runtime_evidence.add_argument("--out", required=True)
    runtime_evidence.add_argument("--config", default=None)
    runtime_evidence.add_argument("--baseline", default=None)

    pack = sub.add_parser("pack", help="Validate and inspect RobotProbe plugin packs.")
    pack_sub = pack.add_subparsers(dest="pack_command", required=True)
    pack_validate = pack_sub.add_parser("validate", help="Validate a plugin pack.")
    pack_validate.add_argument("path")
    pack_validate.add_argument("--allow-hardware-executable", action="store_true")
    pack_list = pack_sub.add_parser("list", help="List pack plugins and challenge refs.")
    pack_list.add_argument("path")
    pack_test = pack_sub.add_parser("test", help="Run static pack conformance checks.")
    pack_test.add_argument("path")
    pack_test.add_argument("--allow-hardware-executable", action="store_true")
    pack_test.add_argument("--allow-code", action="store_true")
    pack_claims = pack_sub.add_parser("claims", help="Render a pack claim card.")
    pack_claims.add_argument("path")
    pack_init = pack_sub.add_parser("init", help="Scaffold a new RobotProbe plugin pack.")
    pack_init.add_argument("path")
    pack_init.add_argument("--id", required=True)
    pack_init.add_argument("--plugin-id", default=None)
    pack_init.add_argument("--plugin-type", default=None)
    pack_init.add_argument("--template", choices=sorted(PACK_TEMPLATES), default=None)
    pack_init.add_argument("--version", default="0.1.0")
    pack_registry = pack_sub.add_parser("registry", help="Build and inspect pack registries.")
    pack_registry_sub = pack_registry.add_subparsers(dest="pack_registry_command", required=True)
    pack_registry_build = pack_registry_sub.add_parser("build", help="Build a pack registry JSON from pack roots.")
    pack_registry_build.add_argument("paths", nargs="+")
    pack_registry_build.add_argument("--out", required=True)
    pack_registry_build.add_argument("--allow-hardware-executable", action="store_true")
    pack_registry_list = pack_registry_sub.add_parser("list", help="List a pack registry JSON.")
    pack_registry_list.add_argument("registry")

    contracts = sub.add_parser("contracts", help="Evaluate RobotProbe contract DSL files.")
    contracts_sub = contracts.add_subparsers(dest="contracts_command", required=True)
    contracts_check = contracts_sub.add_parser("check", help="Check one contract against one trace.")
    contracts_check.add_argument("--contract", required=True)
    contracts_check.add_argument("--trace", required=True)

    export = sub.add_parser("export", help="Export RobotProbe bundles to sidecar formats.")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    export_sidecar_parser = export_sub.add_parser("sidecar", help="Write RobotProbe JSON sidecar.")
    export_sidecar_parser.add_argument("--bundle", required=True)
    export_sidecar_parser.add_argument("--out", required=True)
    export_mcap_parser = export_sub.add_parser("mcap", help="Write RobotProbe bundle as JSON MCAP.")
    export_mcap_parser.add_argument("--bundle", required=True)
    export_mcap_parser.add_argument("--out", required=True)

    import_parser = sub.add_parser("import", help="Import RobotProbe data from external formats.")
    import_sub = import_parser.add_subparsers(dest="import_command", required=True)
    import_mcap_parser = import_sub.add_parser("mcap", help="Import RobotProbe JSON MCAP into a bundle directory.")
    import_mcap_parser.add_argument("--mcap", required=True)
    import_mcap_parser.add_argument("--out", required=True)

    ingest = sub.add_parser("ingest", help="Ingest external robot logs into RobotProbe traces.")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)
    ingest_ros2 = ingest_sub.add_parser("ros2-mcap", help="Ingest a ROS 2 MCAP log into a RobotProbe trace.")
    ingest_ros2.add_argument("--input", required=True)
    ingest_ros2.add_argument("--out", required=True)
    ingest_ros2.add_argument("--run-id", required=True)
    ingest_ros2.add_argument("--target", default="ros2")
    ingest_ros2.add_argument("--scenario", required=True)
    ingest_vla_eval = ingest_sub.add_parser("vla-eval", help="Ingest a vla-eval episode/result artifact.")
    ingest_vla_eval.add_argument("--input", required=True)
    ingest_vla_eval.add_argument("--out", required=True)
    ingest_vla_eval.add_argument("--run-id", required=True)
    ingest_vla_eval.add_argument("--target", default="vla-eval")
    ingest_vla_eval.add_argument("--scenario", required=True)
    ingest_isaac = ingest_sub.add_parser("isaac-lab-arena", help="Ingest an Isaac Lab-Arena exported trace.")
    ingest_isaac.add_argument("--input", required=True)
    ingest_isaac.add_argument("--out", required=True)
    ingest_isaac.add_argument("--run-id", required=True)
    ingest_isaac.add_argument("--target", default="isaac-lab-arena")
    ingest_isaac.add_argument("--scenario", required=True)

    doctor = sub.add_parser("doctor", help="Inspect RobotProbe environment and optional integrations.")
    doctor.add_argument("--pack", default=None)
    doctor.add_argument("--integration", default=None)

    mech = sub.add_parser("mech", help="Optional white-box mechanistic forensics commands.")
    mech_sub = mech.add_subparsers(dest="mech_command", required=True)
    mech_validate = mech_sub.add_parser("validate", help="Validate a MechProbe capture config.")
    mech_validate.add_argument("--capture-config", required=True)
    mech_collect = mech_sub.add_parser("collect", help="Collect activations through a white-box model adapter.")
    mech_collect.add_argument("--target", required=True)
    mech_collect.add_argument("--trace-pair", default=None)
    mech_collect.add_argument("--clean", default=None)
    mech_collect.add_argument("--perturbed", default=None)
    mech_collect.add_argument("--capture-config", required=True)
    mech_collect.add_argument("--out", required=True)
    mech_cache_info = mech_sub.add_parser("cache-info", help="Inspect a file-backed MechProbe activation cache.")
    mech_cache_info.add_argument("--mech-dir", required=True)
    mech_diff = mech_sub.add_parser("diff-activations", help="Diff clean vs perturbed activation captures.")
    mech_diff.add_argument("--mech-dir", required=True)
    mech_diff.add_argument("--out", required=True)
    mech_feature_diff = mech_sub.add_parser("diff-features", help="Diff candidate feature activations.")
    mech_feature_diff.add_argument("--mech-dir", required=True)
    mech_feature_diff.add_argument("--dictionary", required=True)
    mech_feature_diff.add_argument("--out", required=True)
    mech_faithfulness = mech_sub.add_parser("faithfulness", help="Compute offline consequence-faithfulness over paired activations.")
    mech_faithfulness.add_argument("--mech-dir", required=True)
    mech_faithfulness.add_argument("--summary", required=True)
    mech_faithfulness.add_argument("--target-metric", required=True)
    mech_faithfulness.add_argument("--max-k", type=int, default=None)
    mech_faithfulness.add_argument("--out", required=True)
    mech_report = mech_sub.add_parser("report", help="Render a MechProbe report.")
    mech_report.add_argument("--summary", required=True)
    mech_report.add_argument("--activation-diff", required=True)
    mech_report.add_argument("--feature-diff", default=None)
    mech_report.add_argument("--out", required=True)

    incident_pack = sub.add_parser("incident-pack", help="Download and convert public robot incident datasets.")
    incident_sub = incident_pack.add_subparsers(dest="incident_command", required=True)
    incident_download = incident_sub.add_parser("download", help="Download a targeted public incident dataset seed.")
    incident_download_sub = incident_download.add_subparsers(dest="incident_dataset", required=True)
    incident_download_vifailback = incident_download_sub.add_parser("vifailback", help="Download a small ViFailback seed.")
    incident_download_vifailback.add_argument("--out", required=True)
    incident_download_vifailback.add_argument("--task", default="cut_sausage")
    incident_download_vifailback.add_argument("--no-vqa", action="store_true")
    incident_download_vifailback.add_argument("--include-large-hdf5", action="store_true")
    incident_download_vifailback.add_argument("--episode-id", type=int, action="append", default=[])
    incident_download_robofac = incident_download_sub.add_parser("robofac", help="Download a small RoboFAC seed.")
    incident_download_robofac.add_argument("--out", required=True)
    incident_download_robofac.add_argument("--leaf", default="simulation_data/LiftPegUpright-box/view0")
    incident_download_robofac.add_argument("--no-h5", action="store_true")

    incident_convert = incident_sub.add_parser("convert", help="Convert public incident data to RobotProbe traces.")
    incident_convert_sub = incident_convert.add_subparsers(dest="incident_dataset", required=True)
    incident_convert_vifailback = incident_convert_sub.add_parser("vifailback", help="Convert ViFailback annotations.")
    incident_convert_vifailback.add_argument("--annotation", required=True)
    incident_convert_vifailback.add_argument("--out", required=True)
    incident_convert_vifailback.add_argument("--limit", type=int, default=None)
    incident_convert_vifailback.add_argument("--run-prefix", default="vifailback")
    incident_convert_vifailback.add_argument("--target", default="vifailback")
    incident_convert_vifailback.add_argument("--scenario", default=None)
    incident_convert_robofac = incident_convert_sub.add_parser("robofac", help="Convert RoboFAC metadata/H5 episodes.")
    incident_convert_robofac.add_argument("--metadata", required=True)
    incident_convert_robofac.add_argument("--out", required=True)
    incident_convert_robofac.add_argument("--h5", default=None)
    incident_convert_robofac.add_argument("--limit", type=int, default=None)
    incident_convert_robofac.add_argument("--max-steps", type=int, default=None)
    incident_convert_robofac.add_argument("--run-prefix", default="robofac")
    incident_convert_robofac.add_argument("--target", default="robofac")
    incident_convert_robofac.add_argument("--scenario", default=None)

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "summarize":
        result = summarize_trace_pair(
            args.clean,
            args.perturbed,
            intervention_manifest_ref=args.manifest,
            metrics=_parse_csv(args.metrics),
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.command == "report":
        rendered = write_incident_report(
            args.summary,
            check_path=args.check,
            out_path=args.out,
            title=args.title,
        )
        if args.out is None:
            sys.stdout.write(rendered)
        return 0
    if args.command == "validate-trace":
        result = validate_trace_file(args.trace)
        json.dump(dataclasses.asdict(result), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result.valid else 1
    if args.command == "conformance":
        result = check_conformance_file(args.trace, target_level=args.level)
        json.dump(dataclasses.asdict(result), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result.conformant else 1
    if args.command == "campaign":
        if args.campaign_command == "run":
            report = run_campaign_file(args.campaign, out_dir=args.out)
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            if args.fail_on_violation and report["n_faults_violating_contracts"] > 0:
                return 1
            return 0
    if args.command == "worldmodel":
        if args.worldmodel_command == "faithfulness":
            report = check_world_model_faithfulness(
                args.trace,
                state_tolerance=args.state_tolerance,
                divergence_threshold=args.divergence_threshold,
            )
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2, sort_keys=True)
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            if args.fail_on_divergence and report["n_divergent"] > 0:
                return 1
            return 0
    if args.command == "replay":
        result = build_replay_plan(
            args.trace,
            level=args.level,
            action_envelope_path=args.action_envelope,
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.command == "bundle":
        result = create_incident_bundle(
            clean_trace=args.clean,
            perturbed_trace=args.perturbed,
            manifest=args.manifest,
            out_dir=args.out,
            challenge=args.challenge,
            metrics=_parse_csv(args.metrics),
            title=args.title,
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.command == "runtime":
        if args.runtime_command == "inspect":
            result = inspect_runtime_config(args.config)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0 if result["valid"] else 1
        if args.runtime_command == "run":
            result = run_runtime_config(args.config)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0 if result["outcome"]["success"] else 1
        if args.runtime_command == "evidence":
            result = create_runtime_evidence_bundle(
                trace_path=args.trace,
                out_dir=args.out,
                config_path=args.config,
                baseline_path=args.baseline,
            )
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0 if result["runtime_integrity"] == "passed" else 1
    if args.command == "pack":
        if args.pack_command == "validate":
            result = validate_pack(
                args.path,
                allow_hardware_executable=args.allow_hardware_executable,
            )
            json.dump(result.to_dict(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0 if result.valid else 1
        if args.pack_command == "list":
            json.dump(list_pack(args.path), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.pack_command == "test":
            result = test_pack(
                args.path,
                allow_hardware_executable=args.allow_hardware_executable,
                allow_code=args.allow_code,
            )
            json.dump(result.to_dict(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0 if result.valid else 1
        if args.pack_command == "claims":
            sys.stdout.write(build_claim_card(args.path))
            return 0
        if args.pack_command == "init":
            if args.template is None and (args.plugin_id is None or args.plugin_type is None):
                parser.error("--plugin-id and --plugin-type are required without --template")
            result = init_pack(
                args.path,
                pack_id=args.id,
                plugin_id=args.plugin_id,
                plugin_type=args.plugin_type,
                version=args.version,
                template=args.template,
            )
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.pack_command == "registry":
            if args.pack_registry_command == "build":
                result = build_pack_registry(
                    args.paths,
                    out_path=args.out,
                    allow_hardware_executable=args.allow_hardware_executable,
                )
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
            if args.pack_registry_command == "list":
                result = load_pack_registry(args.registry)
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
    if args.command == "contracts":
        if args.contracts_command == "check":
            result = check_contract_file(args.contract, args.trace)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
    if args.command == "export":
        if args.export_command == "sidecar":
            result = export_sidecar(args.bundle, args.out)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.export_command == "mcap":
            result = export_mcap_bundle(args.bundle, args.out)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
    if args.command == "import":
        if args.import_command == "mcap":
            result = import_mcap_bundle(args.mcap, args.out)
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
    if args.command == "ingest":
        if args.ingest_command == "ros2-mcap":
            result = ingest_ros2_mcap(
                args.input,
                args.out,
                run_id=args.run_id,
                target=args.target,
                scenario=args.scenario,
            )
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.ingest_command == "vla-eval":
            result = ingest_vla_eval_episode(
                args.input,
                args.out,
                run_id=args.run_id,
                target=args.target,
                scenario=args.scenario,
            )
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.ingest_command == "isaac-lab-arena":
            result = ingest_isaac_lab_arena_trace(
                args.input,
                args.out,
                run_id=args.run_id,
                target=args.target,
                scenario=args.scenario,
            )
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
    if args.command == "incident-pack":
        if args.incident_command == "download":
            if args.incident_dataset == "vifailback":
                result = download_vifailback_seed(
                    args.out,
                    task=args.task,
                    include_vqa=not args.no_vqa,
                    include_large_hdf5=args.include_large_hdf5,
                    episode_ids=args.episode_id,
                )
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
            if args.incident_dataset == "robofac":
                result = download_robofac_seed(
                    args.out,
                    leaf=args.leaf,
                    include_h5=not args.no_h5,
                )
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
        if args.incident_command == "convert":
            if args.incident_dataset == "vifailback":
                result = convert_vifailback_annotations(
                    args.annotation,
                    args.out,
                    limit=args.limit,
                    run_prefix=args.run_prefix,
                    target=args.target,
                    scenario=args.scenario,
                )
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
            if args.incident_dataset == "robofac":
                result = convert_robofac_episodes(
                    args.metadata,
                    args.out,
                    h5_path=args.h5,
                    limit=args.limit,
                    max_steps=args.max_steps,
                    run_prefix=args.run_prefix,
                    target=args.target,
                    scenario=args.scenario,
                )
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                return 0
    if args.command == "doctor":
        result = run_doctor(pack_path=args.pack, integration=args.integration)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result["ok"] else 1
    if args.command == "mech":
        try:
            return _cmd_mech(args)
        except (OSError, ValueError) as exc:
            json.dump(
                {
                    "format": "robotprobe.mech.error.v0",
                    "status": "error",
                    "command": args.mech_command,
                    "error": str(exc),
                },
                sys.stdout,
                indent=2,
                sort_keys=True,
            )
            sys.stdout.write("\n")
            return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_mech(args: argparse.Namespace) -> int:
    if args.mech_command == "validate":
        result = validate_capture_config(load_capture_config(args.capture_config))
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result["valid"] else 1
    if args.mech_command == "collect":
        result = collect_activations(
            target=args.target,
            trace_pair=args.trace_pair,
            clean_trace=args.clean,
            perturbed_trace=args.perturbed,
            capture_config=args.capture_config,
            out_dir=args.out,
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result.get("status") == "captured" else 2
    if args.mech_command == "cache-info":
        result = build_cache_summary(args.mech_dir)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.mech_command == "diff-activations":
        result = diff_activations(mech_dir=args.mech_dir, out_path=args.out)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.mech_command == "diff-features":
        result = diff_features(mech_dir=args.mech_dir, dictionary_path=args.dictionary, out_path=args.out)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.mech_command == "faithfulness":
        result = compute_consequence_faithfulness(
            mech_dir=args.mech_dir,
            summary_path=args.summary,
            target_metric=args.target_metric,
            max_k=args.max_k,
            out_path=args.out,
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.mech_command == "report":
        result = write_mech_report(
            summary_path=args.summary,
            activation_diff_path=args.activation_diff,
            feature_diff_path=args.feature_diff,
            out_path=args.out,
        )
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    raise ValueError(f"unknown MechProbe command: {args.mech_command}")


DEMO_CHALLENGE = "robotprobe-v0-gripper-freeze"
DEMO_TITLE = "Quickstart: Gripper Freeze"
DEMO_METRICS = ("action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss")

# Marker written into the output directory. `demo` only ever clears a directory
# it can prove is its own previous output — clearing an arbitrary user-supplied
# path is a destructive-action bug (see issue #40).
_DEMO_MARKER = ".robotprobe-demo-out"
_DEMO_KNOWN_ENTRIES = {
    _DEMO_MARKER, "report.md", "bundle.json", "bundle.command.json", "summary.json",
    "causal_diff.json", "check.json", "clean.validation.json", "perturbed.validation.json",
    "replay_l0.json", "traces", "manifests",
}

# (packaged path, source-tree path). The packaged copy is what `pip install`
# ships; the source path keeps a checkout working if package data was not
# rebuilt. tests/test_demo_command.py pins the two copies together so they
# cannot silently drift.
_DEMO_ASSETS = {
    "clean": ("data/demo/clean.jsonl", "examples/quickstart/traces/clean.jsonl"),
    "perturbed": ("data/demo/gripper_freeze_mid.jsonl", "examples/quickstart/traces/gripper_freeze_mid.jsonl"),
    "manifest": ("data/demo/intervention.yaml", "interventions/gripper_freeze_mid.yaml"),
}


def _demo_asset(name: str) -> Path:
    packaged_rel, source_rel = _DEMO_ASSETS[name]
    package_root = Path(__file__).resolve().parent
    packaged = package_root / packaged_rel
    if packaged.is_file():
        return packaged
    from_source = package_root.parent / source_rel
    if from_source.is_file():
        return from_source
    raise SystemExit(
        f"demo asset {name!r} not found. Looked for:\n  {packaged}\n  {from_source}\n"
        "Reinstall robotprobe, or run this from a source checkout."
    )


def _prepare_demo_out(out: Path) -> None:
    """Create the demo output directory, clearing only our own previous output.

    A caller-supplied `--out` is untrusted: `--out .` or `--out ~` is a plausible
    slip while following the docs, and recursively emptying it would destroy the
    tree before anything is written. So we clear the directory only when every
    entry is a known demo artifact, and refuse otherwise.
    """
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"--out {out} exists and is not a directory.")
        entries = list(out.iterdir())
        foreign = sorted(p.name for p in entries if p.name not in _DEMO_KNOWN_ENTRIES)
        if foreign:
            preview = ", ".join(foreign[:5]) + (" …" if len(foreign) > 5 else "")
            raise SystemExit(
                f"refusing to write into {out}: it is not empty and was not produced by "
                f"`robotprobe demo`.\n  Unexpected entries: {preview}\n"
                f"Use an empty directory, or remove it yourself."
            )
        for child in entries:
            if child.name == _DEMO_MARKER:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / _DEMO_MARKER).write_text(
        "Output of `robotprobe demo`. Safe to delete; regenerated on each run.\n", encoding="utf-8"
    )


def _cmd_demo(args) -> int:
    """Run the bundled quickstart incident end to end and write the artifacts.

    This is the whole point of a first run: one command, no paths to supply, a
    real incident report at the end. It calls the same functions the individual
    subcommands call, so it cannot drift from them.
    """
    out = Path(args.out).resolve()
    _prepare_demo_out(out)

    clean = _demo_asset("clean")
    perturbed = _demo_asset("perturbed")
    manifest = _demo_asset("manifest")

    for label, trace in (("clean", clean), ("perturbed", perturbed)):
        validation = validate_trace_file(trace)
        (out / f"{label}.validation.json").write_text(
            json.dumps(dataclasses.asdict(validation), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    bundle = create_incident_bundle(
        clean_trace=clean,
        perturbed_trace=perturbed,
        manifest=manifest,
        out_dir=out,
        challenge=DEMO_CHALLENGE,
        metrics=list(DEMO_METRICS),
        title=DEMO_TITLE,
    )
    (out / "bundle.command.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    replay = build_replay_plan(perturbed, level="L0")
    (out / "replay_l0.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"RobotProbe quickstart demo wrote {out}")
    print(f"Open {out / 'report.md'} for the incident report.")
    return 0


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
