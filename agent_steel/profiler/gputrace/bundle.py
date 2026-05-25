"""Top-level enumeration of files in a .gputrace bundle and the
``parse(path) -> dict`` entry point.

A .gputrace bundle is a directory. Observed files (macOS 15/26 Xcode
17, single-frame compute capture):

  capture                          MTSP record stream (binary)
  unsorted-capture                 secondary MTSP stream
  index                            unknown offset index (binary)
  metadata                         Apple binary plist (capture session info)
  store0                           zlib-compressed sparse storage
  CC89742E83E502BB                 embedded MTLLibrary (.metallib)
  device-resources-0x<hex>         MTSP record stream of pre-existing resources
  delta-device-resources-0x<hex>   MTSP stream of newly-created resources
  unused-device-resources-0x<hex>  resources not used in the capture
  startup-0-platform               capture process info (very short)
  startup-1-platform               capture process info (very short)
"""
from __future__ import annotations

import os
from typing import Any

from . import capture as _capture
from . import metadata as _metadata


def list_files(path: str) -> dict[str, dict]:
    """Return {filename: {size, kind}} for each file in the bundle dir."""
    out: dict[str, dict] = {}
    if not os.path.isdir(path):
        return out
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        kind = _classify(name, full)
        out[name] = {"size": size, "kind": kind}
    return out


def _classify(name: str, full_path: str) -> str:
    if name == "metadata":
        return "bplist"
    if name == "capture" or name == "unsorted-capture":
        return "mtsp_capture"
    if name.startswith("device-resources-") or name.startswith("delta-device-resources-") or name.startswith("unused-device-resources-"):
        return "mtsp_resources"
    if name == "index":
        return "index"
    if name == "store0":
        return "zlib_store"
    if name.startswith("startup-"):
        return "platform_info"
    # else sniff magic
    try:
        with open(full_path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unknown"
    if head[:4] == b"MTLB":
        return "metallib"
    if head[:4] == b"MTSP":
        return "mtsp_capture"
    return "unknown"


def parse(path: str) -> dict[str, Any]:
    """Parse a .gputrace bundle and return a profile-oriented dict.

    See REPORT.md for the schema and confidence levels.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"not a .gputrace bundle dir: {path}")

    files = list_files(path)
    md = _metadata.parse_metadata(path)
    cap = _capture.parse_capture(path)

    # Device summary from metadata.
    device = {
        "device_id": md.get("DYCaptureSession.deviceId"),
        "graphics_api": md.get("DYCaptureSession.graphics_api"),
        "capture_version": md.get("DYCaptureSession.capture_version"),
        "metal_link_version": md.get("DYCaptureSession.library_link_time_versions", {}).get("Metal"),
        "captured_frames": md.get("DYCaptureEngine.captured_frames_count"),
    }

    # Per-bundle metallib name (if present).
    metallib = None
    for fname, info in files.items():
        if info["kind"] == "metallib":
            metallib = {"name": fname, "size": info["size"]}
            break

    return {
        "bundle_path": os.path.abspath(path),
        "device": device,
        "metallib": metallib,
        "files": files,
        "command_buffers": cap.get("command_buffers", []),
        "_diagnostics": {
            "record_counts": cap.get("_record_counts", {}),
            "unknown_record_types": cap.get("_unknown_record_types", {}),
        },
    }
