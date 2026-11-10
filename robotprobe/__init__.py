"""RobotProbe: trace, intervention, and causal-diff primitives."""

from robotprobe.attach import PolicyProbe, Watchpoint
from robotprobe.attacks import (
    ActionFaultAttack,
    Attack,
    InstructionOverrideAttack,
    NoAttack,
    Observation,
    StaleObservationReplayAttack,
)
from robotprobe.conformance import check_conformance_file
from robotprobe.campaign import run_campaign_file
from robotprobe.worldmodel import check_world_model_faithfulness
from robotprobe.core import InterventionManifest, ReplayLevel, load_intervention_manifest
from robotprobe.record import TraceRecorder, TraceSpec
from robotprobe.replay import build_replay_plan
from robotprobe.schema import validate_trace_file
from robotprobe.summarize import summarize_trace_pair

__all__ = [
    "ActionFaultAttack",
    "Attack",
    "InstructionOverrideAttack",
    "InterventionManifest",
    "NoAttack",
    "PolicyProbe",
    "Observation",
    "ReplayLevel",
    "StaleObservationReplayAttack",
    "Watchpoint",
    "TraceRecorder",
    "TraceSpec",
    "build_replay_plan",
    "check_conformance_file",
    "check_world_model_faithfulness",
    "run_campaign_file",
    "load_intervention_manifest",
    "summarize_trace_pair",
    "validate_trace_file",
]
