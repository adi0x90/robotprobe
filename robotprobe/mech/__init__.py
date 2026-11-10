"""Optional white-box mechanistic forensics primitives for RobotProbe.

MechProbe stays incident-first: it links selected internal activations to RobotProbe traces
without making SAE labels or causal feature claims by default.
"""

from robotprobe.mech.cache_view import ActivationCacheView, ActivationSequence
from robotprobe.mech.capabilities import MechAdapterCapabilities
from robotprobe.mech.schemas import MechEvidenceLevel, ModelActivationEvent

__all__ = [
    "ActivationCacheView",
    "ActivationSequence",
    "MechAdapterCapabilities",
    "MechEvidenceLevel",
    "ModelActivationEvent",
]
