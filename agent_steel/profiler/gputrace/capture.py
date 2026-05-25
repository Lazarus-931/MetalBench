"""Parse the ``capture`` (and ``unsorted-capture``) binary stream in a
.gputrace bundle.

Format (reverse-engineered):

    magic: 4 bytes b"MTSP"
    version: u32  (observed 0x00000004)
    records: stream of [size:u32][type:u32][payload of size-8 bytes]

Each record payload begins with 24 bytes of (mostly-zero) header — likely
reserved for sequence number / sample slot, but in single-frame captures
this is always zero. Then a length-prefixed Objective-C-style type
signature, padded so the next field is 8-byte aligned from the start of
the payload. The trailing bytes of every record contain ~24 bytes that
look like backtrace return addresses (unused by us).

We only decode the record types relevant for compute profiling:

    0xffffc05e  newComputePipelineState — carries the function name
    0xffffc00c  resource (MTLBuffer) — carries label and handle
    0xffffc013  command buffer — carries label
    0xffffc02d  compute command encoder — carries label
    0xffffc030  setBuffer:(buffer, offset, index)
    0xffffc132  dispatchThreads:threadsPerThreadgroup: — grid + tg
    0xffffc017  commit
    0xffffc01d  waitUntilCompleted
    0xffffc03b  endEncoding
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Any


RT_PIPELINE        = 0xffffc05e
RT_RESOURCE        = 0xffffc00c    # labelObject (sets MTLBuffer label)
RT_NEW_BUFFER      = 0xffffc046    # newBufferWithLength:options:
RT_CB              = 0xffffc013
RT_ENCODER         = 0xffffc02d
RT_SET_BUFFER      = 0xffffc030
RT_DISPATCH        = 0xffffc132
RT_COMMIT          = 0xffffc017
RT_WAIT            = 0xffffc01d
RT_END_ENCODING    = 0xffffc03b


RECORD_NAMES = {
    RT_PIPELINE: "computePipelineState",
    RT_RESOURCE: "labelObject",
    RT_NEW_BUFFER: "newBufferWithLength",
    RT_CB: "commandBuffer",
    RT_ENCODER: "computeCommandEncoder",
    RT_SET_BUFFER: "setBuffer",
    RT_DISPATCH: "dispatchThreads",
    RT_COMMIT: "commit",
    RT_WAIT: "waitUntilCompleted",
    RT_END_ENCODING: "endEncoding",
}


@dataclass
class Record:
    offset: int
    size: int
    type: int
    payload: bytes


def iter_records(data: bytes) -> list[Record]:
    """Walk the MTSP record stream, skipping the 8-byte file header."""
    if data[:4] != b"MTSP":
        raise ValueError("not an MTSP capture stream")
    off = 8
    recs: list[Record] = []
    while off + 8 <= len(data):
        size, typ = struct.unpack_from("<II", data, off)
        if size < 8 or size > len(data) - off:
            break
        recs.append(Record(off, size, typ, data[off + 8:off + size]))
        off += size
    return recs


def _read_cstring(buf: bytes, offset: int, maxlen: int = 256) -> str:
    end = buf.find(b"\x00", offset, offset + maxlen)
    if end < 0:
        end = min(offset + maxlen, len(buf))
    return buf[offset:end].decode("utf-8", errors="replace")


def _parse_type_encoded(payload: bytes, align: int = 4) -> tuple[str, int]:
    """Return (type-encoding string, byte-offset where typed-field area starts).

    Records share a preamble:
        [24 bytes reserved=0]
        [4 bytes "tag" (u32) — class ID or type-encoded length, not a string length]
        [type-encoding ASCII string, NUL-terminated, padded to ``align``]
        [typed field area...]

    ``align`` is record-type-dependent (observed values: 4 for setBuffer,
    8 for dispatchThreads). Caller passes the right value.
    """
    if len(payload) < 28:
        return "", 28
    te_off = 28
    te_end = payload.find(b"\x00", te_off)
    if te_end < 0:
        return "", 28
    te = payload[te_off:te_end].decode("ascii", errors="replace")
    field_start = ((te_end + 1) + align - 1) & ~(align - 1)
    return te, field_start


def parse_pipeline(rec: Record) -> dict:
    # function name is a NUL-terminated C-string starting at payload offset 40
    name = _read_cstring(rec.payload, 40)
    return {"_kind": "pipeline", "function": name}


def parse_new_buffer(rec: Record) -> dict | None:
    """Parse newBufferWithLength:options: — te='Culul'.

    Field area (4-byte aligned): device-handle(u64), length(u64), options(u64).
    The returned buffer handle isn't directly here; it's referenced by the
    immediately-following labelObject (c00c) record.
    """
    te, fs = _parse_type_encoded(rec.payload, align=4)
    if not te.startswith("Cul"):
        return None
    pl = rec.payload
    if fs + 24 > len(pl):
        return None
    device = struct.unpack_from("<Q", pl, fs)[0]
    length = struct.unpack_from("<Q", pl, fs + 8)[0]
    options = struct.unpack_from("<Q", pl, fs + 16)[0]
    return {"_kind": "newBuffer", "device": device, "length": length, "options": options}


def parse_resource(rec: Record) -> dict:
    # label is a NUL-terminated C-string starting at offset 40
    # handle (object pointer) at offset 32 (8 bytes)
    label = _read_cstring(rec.payload, 40)
    handle = 0
    if len(rec.payload) >= 40:
        handle = struct.unpack_from("<Q", rec.payload, 32)[0]
    return {"_kind": "resource", "label": label, "handle": handle}


def parse_cb(rec: Record) -> dict:
    label = _read_cstring(rec.payload, 40)
    return {"_kind": "commandBuffer", "label": label}


def parse_encoder(rec: Record) -> dict:
    label = _read_cstring(rec.payload, 40)
    return {"_kind": "encoder", "label": label}


def parse_set_buffer(rec: Record) -> dict | None:
    # te = "Ctulul" -> char, ptr(encoder), ulong(buffer-handle), ulong(offset), ulong(index)
    te, fs = _parse_type_encoded(rec.payload, align=4)
    if not te.startswith("Ctul"):
        return None
    pl = rec.payload
    if fs + 32 > len(pl):
        return None
    encoder = struct.unpack_from("<Q", pl, fs)[0]
    buffer_h = struct.unpack_from("<Q", pl, fs + 8)[0]
    offset = struct.unpack_from("<Q", pl, fs + 16)[0]
    index = struct.unpack_from("<Q", pl, fs + 24)[0]
    return {
        "_kind": "setBuffer",
        "encoder": encoder,
        "buffer": buffer_h,
        "offset": offset,
        "index": index,
    }


def parse_dispatch(rec: Record) -> dict | None:
    # te = "C@3ul@3ul" -> char, MTLSize grid (3 u64), MTLSize tg (3 u64)
    # Field layout after te-padding: encoder handle(u64), grid(3*u64), tg(3*u64)
    te, fs = _parse_type_encoded(rec.payload, align=8)
    if "3ul" not in te:
        return None
    pl = rec.payload
    if fs + 8 + 48 > len(pl):
        return None
    encoder = struct.unpack_from("<Q", pl, fs)[0]
    grid = list(struct.unpack_from("<3Q", pl, fs + 8))
    tg = list(struct.unpack_from("<3Q", pl, fs + 8 + 24))
    return {"_kind": "dispatchThreads", "encoder": encoder, "grid": grid, "threadgroup": tg}


def parse_capture(path: str) -> dict:
    """Parse capture + unsorted-capture and assemble a logical view.

    Strategy: stream both files in order, accumulate handles -> labels,
    and bind dispatch events to the most recently created encoder /
    pipeline state. This is a heuristic that holds when one frame
    contains a small number of encoders, which is the common case for
    single-dispatch captures.
    """
    out: dict[str, Any] = {
        "command_buffers": [],
        "_record_counts": {},
        "_unknown_record_types": {},
    }

    # Prefer "capture" (ordered) over "unsorted-capture" (same logical content
    # in observed bundles). If only unsorted exists, fall back to it.
    sources = []
    cpath = os.path.join(path, "capture")
    upath = os.path.join(path, "unsorted-capture")
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            sources.append(("capture", f.read()))
    elif os.path.exists(upath):
        with open(upath, "rb") as f:
            sources.append(("unsorted-capture", f.read()))

    # Also pull resource definitions from device-resources-* (pre-existing
    # MTLBuffers may live there rather than in capture).
    for name in sorted(os.listdir(path)):
        if name.startswith("device-resources-") or name.startswith("delta-device-resources-"):
            fpath = os.path.join(path, name)
            try:
                with open(fpath, "rb") as f:
                    head = f.read(4)
                    if head == b"MTSP":
                        f.seek(0)
                        sources.append((name, f.read()))
            except OSError:
                pass

    if not sources:
        return out

    # Per-bundle tables
    resources: dict[int, dict] = {}   # handle -> {label, length?, options?}
    last_pipeline: str | None = None
    current_cb: dict | None = None
    current_encoder: dict | None = None
    pending_dispatch_bufs: list[dict] = []
    pending_new_buffer_length: int | None = None
    pending_new_buffer_options: int | None = None

    counts: dict[str, int] = {}
    unknown: dict[int, int] = {}

    def bump(typ_str: str) -> None:
        counts[typ_str] = counts.get(typ_str, 0) + 1

    for src_name, data in sources:
        try:
            recs = iter_records(data)
        except ValueError:
            continue

        for rec in recs:
            name = RECORD_NAMES.get(rec.type)
            if name is None:
                unknown[rec.type] = unknown.get(rec.type, 0) + 1
                continue
            bump(name)

            if rec.type == RT_PIPELINE:
                info = parse_pipeline(rec)
                last_pipeline = info["function"]

            elif rec.type == RT_NEW_BUFFER:
                info = parse_new_buffer(rec)
                if info is not None:
                    # stash the length so the next labelObject can pick it up.
                    pending_new_buffer_length = info["length"]
                    pending_new_buffer_options = info["options"]

            elif rec.type == RT_RESOURCE:
                info = parse_resource(rec)
                if pending_new_buffer_length is not None:
                    info["length"] = pending_new_buffer_length
                    info["options"] = pending_new_buffer_options
                    pending_new_buffer_length = None
                    pending_new_buffer_options = None
                resources[info["handle"]] = info

            elif rec.type == RT_CB:
                info = parse_cb(rec)
                current_cb = {"label": info["label"], "dispatches": []}
                out["command_buffers"].append(current_cb)

            elif rec.type == RT_ENCODER:
                info = parse_encoder(rec)
                current_encoder = {"label": info["label"]}
                pending_dispatch_bufs = []

            elif rec.type == RT_SET_BUFFER:
                info = parse_set_buffer(rec)
                if info is not None:
                    bufrec = resources.get(info["buffer"], {})
                    pending_dispatch_bufs.append({
                        "index": info["index"],
                        "offset": info["offset"],
                        "label": bufrec.get("label"),
                        "length": bufrec.get("length"),
                        "options": bufrec.get("options"),
                        "handle": info["buffer"],
                    })

            elif rec.type == RT_DISPATCH:
                info = parse_dispatch(rec)
                if info is None:
                    continue
                disp = {
                    "function": last_pipeline,
                    "grid": info["grid"],
                    "threadgroup": info["threadgroup"],
                    "encoder_label": current_encoder["label"] if current_encoder else None,
                    "buffers": list(pending_dispatch_bufs),
                }
                if current_cb is None:
                    current_cb = {"label": None, "dispatches": []}
                    out["command_buffers"].append(current_cb)
                current_cb["dispatches"].append(disp)
                pending_dispatch_bufs = []

            elif rec.type == RT_END_ENCODING:
                current_encoder = None
                pending_dispatch_bufs = []
            # RT_COMMIT, RT_WAIT: no payload we need

    out["_record_counts"] = counts
    out["_unknown_record_types"] = {hex(k): v for k, v in unknown.items()}
    return out
