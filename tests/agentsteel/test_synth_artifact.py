"""Anchor test: parse the .gputrace → run the chip-aware synthesizer → check
that the key SOL columns are in the same ballpark as Xcode's CSV ground truth.

Pure-Python. No GPU, no LLM."""
from __future__ import annotations
import csv
from pathlib import Path

from agent_steel.profiler.chip_metrics import derive_metrics
from agent_steel.profiler.gputrace import parse

ART = Path(__file__).parent / "artifacts"


def _synth() -> dict:
    return derive_metrics(
        bench={
            "kernel_ms": 0.014,
            "flops": 18.2e9 * 0.014e-3,
            "bytes": 145.9e9 * 0.014e-3,
            "detected_gpu_cores": 8,
        },
        parsed_trace=parse(str(ART / "relu.gputrace")),
        generation="m2", variant="base",
    )


def _csv_row() -> dict:
    with (ART / "relu.csv").open() as f:
        return next(csv.DictReader(f))


def test_parser_extracts_dispatch():
    trace = parse(str(ART / "relu.gputrace"))
    dispatches = [d for cb in trace["command_buffers"] for d in cb["dispatches"]]
    assert len(dispatches) == 1
    assert dispatches[0]["grid"] == [65536, 1, 1]
    assert dispatches[0]["threadgroup"] == [1024, 1, 1]


def test_synth_returns_xcode_columns():
    synth = _synth()["synth"]
    for k in ("alu_utilization_pct", "kernel_occupancy_pct",
              "device_memory_bandwidth_pct", "kernel_alu_instructions",
              "bytes_read_from_device_memory_bytes", "gpu_time_ns"):
        assert k in synth


def test_synth_bytes_read_matches_csv():
    synth = _synth()["synth"]
    sv = float(synth["bytes_read_from_device_memory_bytes"])
    cv = float(_csv_row()["Bytes Read From Device Memory"])
    assert abs(sv - cv) / cv <= 0.10
