#!/usr/bin/env python3
"""MetalBench environment bootstrap.

Run this once after cloning:
    python3 setup.py

Checks (and tries to fix):
    1. macOS + Apple Silicon
    2. Xcode developer tools     (xcode-select -p)
    3. Metal Toolchain           (xcrun -sdk macosx metal -v)  ← the usual blocker
    4. Python deps               (pip install -r requirements.txt)
    5. Host binary builds        (make host)
    6. Chip detection works      (mlx_helpers.chip_info)

Exit code 0 = ready to bench. Anything else = something needed your attention
and the script told you what.
"""
from __future__ import annotations
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def step(name: str) -> None:
    print(f"\n{BOLD}== {name} =={RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --- 1. Platform -------------------------------------------------------------

def check_platform() -> bool:
    step("platform")
    if platform.system() != "Darwin":
        fail(f"requires macOS, found {platform.system()}")
        return False
    if platform.machine() != "arm64":
        warn(f"Apple Silicon recommended, found {platform.machine()}")
    ok(f"{platform.system()} {platform.release()} on {platform.machine()}")
    return True


# --- 2. Xcode CLT ------------------------------------------------------------

def check_xcode() -> bool:
    step("Xcode developer tools")
    p = run(["xcode-select", "-p"])
    if p.returncode != 0 or not p.stdout.strip():
        fail("Xcode developer tools not found")
        print("    fix: xcode-select --install   (or install Xcode.app)")
        return False
    ok(f"developer dir: {p.stdout.strip()}")
    return True


# --- 3. Metal Toolchain ------------------------------------------------------

def check_metal_toolchain() -> bool:
    step("Metal toolchain")
    p = run(["xcrun", "-sdk", "macosx", "metal", "-v"])
    out = (p.stdout + p.stderr).lower()
    if "missing metal toolchain" in out or "cannot execute tool 'metal'" in out:
        warn("Metal toolchain not installed — attempting download")
        print(f"    running: {BOLD}xcodebuild -downloadComponent MetalToolchain{RESET}")
        print(f"    (~few hundred MB; may take a while; you may need to authenticate)")
        try:
            r = subprocess.run(
                ["xcodebuild", "-downloadComponent", "MetalToolchain"],
                check=False,
            )
        except KeyboardInterrupt:
            fail("download interrupted")
            return False
        if r.returncode != 0:
            fail("download failed")
            print("    fix: run the command above manually and re-run setup.py")
            return False
        # re-check
        p = run(["xcrun", "-sdk", "macosx", "metal", "-v"])
        if "missing metal toolchain" in (p.stdout + p.stderr).lower():
            fail("toolchain still missing after download")
            return False
        ok("Metal toolchain installed")
        return True
    if p.returncode != 0:
        fail(f"unexpected metal error: {p.stderr.strip()[:200]}")
        return False
    ok(p.stderr.splitlines()[0].strip() if p.stderr else "metal compiler available")
    return True


# --- 4. Python deps ----------------------------------------------------------

def check_python_deps() -> bool:
    step("Python dependencies")
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        fail("requirements.txt missing")
        return False
    print(f"    pip install -r {req.name}")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        check=False,
    )
    if r.returncode != 0:
        fail("pip install failed")
        return False
    # quick sanity import
    try:
        import mlx.core  # noqa: F401
        import numpy     # noqa: F401
        import pydantic  # noqa: F401
        ok("mlx, numpy, pydantic importable")
    except ImportError as e:
        fail(f"import after install failed: {e}")
        return False
    return True


# --- 5. Host build -----------------------------------------------------------

def check_host_build() -> bool:
    step("host binary")
    if not shutil.which("make"):
        fail("`make` not on PATH")
        return False
    r = subprocess.run(["make", "host"], cwd=REPO_ROOT)
    if r.returncode != 0:
        fail("`make host` failed")
        return False
    bin_path = REPO_ROOT / "build" / "metalbench_host"
    if not bin_path.exists():
        fail(f"binary missing after build: {bin_path}")
        return False
    ok(f"built {bin_path.relative_to(REPO_ROOT)}")
    return True


# --- 6. Chip detection -------------------------------------------------------

def check_chip() -> bool:
    step("chip detection")
    sys.path.insert(0, str(REPO_ROOT / "mlx" / "scripts"))
    try:
        import mlx_helpers as H  # type: ignore
        info = H.chip_info()
    except Exception as e:
        fail(f"chip_info() raised: {e!r}")
        return False
    if info["type"] == "unknown":
        warn(f"chip type unrecognized — bucket will be '{info['bucket']}'")
    else:
        ok(f"{info['type']}  '{info['name']}'  bucket='{info['bucket']}'  "
           f"cpu={info['cpu_cores']}  gpu={info['gpu_cores']}  "
           f"ram={info['ram_bytes']/1e9:.0f}GB")
    return True


# --- main --------------------------------------------------------------------

def main() -> int:
    print(f"{BOLD}MetalBench setup{RESET}")
    checks = [
        check_platform,
        check_xcode,
        check_metal_toolchain,
        check_python_deps,
        check_host_build,
        check_chip,
    ]
    failures = [c.__name__ for c in checks if not c()]
    print()
    if failures:
        fail(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    # Echo the chip info one more time so it's the LAST line before "complete".
    sys.path.insert(0, str(REPO_ROOT / "mlx" / "scripts"))
    try:
        import mlx_helpers as H  # type: ignore
        c = H.chip_info()
        print(
            f"\n{BOLD}device:{RESET} {c['name']} ({c['type']}) | "
            f"{c['cpu_cores']} CPU | {c['gpu_cores']} GPU cores | "
            f"{c['ram_bytes']/1e9:.0f} GB"
        )
    except Exception:
        pass
    print(f"{GREEN}{BOLD}setup complete.{RESET} run a benchmark with: ./bench <kernel-name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
