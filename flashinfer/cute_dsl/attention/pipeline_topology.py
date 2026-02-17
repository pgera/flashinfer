"""PipelineTopology — stub for declarative pipeline graph.

Will replace the imperative pipeline creation code (~80 lines of
make_pipeline_participants calls) with a declarative graph that can be
validated, visualized, and swapped between kernel variants.

Current pipeline setup lives in:
- prefill.py: lines 824-899 (10 pipelines)
- mla.py: lines 850-870 (5 pipelines)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type


@dataclass
class PipelineSpec:
    """Describes a single pipeline in the topology."""

    name: str
    pipeline_type: str  # "PipelineTmaUmma", "PipelineUmmaAsync", "PipelineAsync"
    stages: int
    producer_role: str  # e.g., "loader", "mma", "softmax0"
    consumer_role: str  # e.g., "mma", "softmax0", "correction"
    tx_count: int = 0
