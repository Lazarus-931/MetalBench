"""Parse the ``metadata`` binary plist in a .gputrace bundle."""
import plistlib
import os


def parse_metadata(path: str) -> dict:
    """Load metadata bplist. Returns {} if missing."""
    mpath = os.path.join(path, "metadata")
    if not os.path.exists(mpath):
        return {}
    with open(mpath, "rb") as f:
        try:
            return plistlib.load(f)
        except Exception as e:  # noqa: BLE001
            return {"_parse_error": str(e)}
