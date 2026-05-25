"""JSONL-backed attempt database.

One file per (kernel, chip) pair. Append-only. Cheap to read end-to-end
(typical attempt counts are 5-50; even a chatty agent run rarely exceeds
500 entries per file).
"""
from __future__ import annotations
import json
from pathlib import Path

from .models import AttemptEntry

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = REPO / ".agent-steel" / "history"


class AttemptDB:
    def __init__(self, root: Path | None = None):
        self.root = root or DEFAULT_HISTORY_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kernel: str, chip: str) -> Path:
        # Sanitize: chip is already "apple-mN", kernel is a registry name.
        safe = f"{kernel}__{chip.replace('/', '_').replace(' ', '_')}.jsonl"
        return self.root / safe

    # --- write ---

    def append(self, entry: AttemptEntry) -> None:
        """Append one attempt to the right file. Creates the file if needed."""
        path = self._path(entry.kernel, entry.chip)
        with path.open("a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    # --- read ---

    def read(self, kernel: str, chip: str) -> list[AttemptEntry]:
        """Return all attempts for (kernel, chip), oldest first."""
        path = self._path(kernel, chip)
        if not path.is_file():
            return []
        out: list[AttemptEntry] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Tolerate forward-compatible schema by dropping unknown fields.
                    known = set(AttemptEntry.__dataclass_fields__.keys())
                    out.append(AttemptEntry(**{k: v for k, v in d.items() if k in known}))
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    # --- derived queries (the convenience the agents actually call) ---

    def techniques_tried(self, kernel: str, chip: str) -> list[str]:
        """All techniques attempted on this (kernel, chip), regardless of outcome.

        The Implementor passes this as `prior_attempts` so it skips
        already-tried strategies.
        """
        return [e.technique for e in self.read(kernel, chip) if e.technique]

    def failed_techniques(self, kernel: str, chip: str) -> list[tuple[str, str]]:
        """Techniques that were rejected — (technique, reason) pairs.

        Useful for the Implementor's prompt to negative-bias the LLM
        against re-trying a known-bad approach.
        """
        return [
            (e.technique, e.rollback_reason)
            for e in self.read(kernel, chip)
            if not e.kept and e.technique
        ]

    def best(self, kernel: str, chip: str) -> AttemptEntry | None:
        top = self.top_n_by_time(kernel, chip, n=1, kept_only=True)
        return top[0] if top else None

    def top_n_by_time(
        self, kernel: str, chip: str, n: int = 5, kept_only: bool = True,
    ) -> list[AttemptEntry]:
        """Return up to N attempts ordered by after_ms ascending (fastest first).

        Entries with no after_ms are skipped. If kept_only is True, only
        attempts the Verifier accepted are returned.
        """
        entries = [
            e for e in self.read(kernel, chip)
            if isinstance(e.after_ms, (int, float))
        ]
        if kept_only:
            entries = [e for e in entries if e.kept]
        entries.sort(key=lambda e: e.after_ms)  # type: ignore[arg-type, return-value]
        return entries[:n]
