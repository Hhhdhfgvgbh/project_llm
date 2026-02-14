from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class StageMeta:
    stage_id: str
    version: int
    parent_version: int | None
    timestamp: str
    input_hash: str
    config_hash: str


@dataclass
class SessionSummary:
    id: str
    created_at: str
    mode: str
    stages: list[str]


@dataclass
class SessionDetails:
    id: str
    final_output: str
    stages: dict[str, dict[str, Any]]


class SessionManager:
    def __init__(self, root: Path | str = "sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> Path:
        session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        target = self.root / session_id
        target.mkdir(parents=True, exist_ok=False)
        return target

    def write_stage_result(
        self,
        session_path: Path,
        stage_id: str,
        version: int,
        parent_version: int | None,
        input_payload: dict[str, Any],
        config_payload: dict[str, Any],
        output_payload: dict[str, Any],
    ) -> Path:
        stage_dir = session_path / f"{stage_id}_v{version}"
        stage_dir.mkdir(parents=True, exist_ok=False)

        meta = StageMeta(
            stage_id=stage_id,
            version=version,
            parent_version=parent_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash=self._stable_hash(input_payload),
            config_hash=self._stable_hash(config_payload),
        )

        (stage_dir / "meta.json").write_text(
            json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (stage_dir / "input.json").write_text(
            json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (stage_dir / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (stage_dir / "output.json").write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return stage_dir

    def write_final_output(self, session_path: Path, final_output: str, mode: str) -> Path:
        (session_path / "final.txt").write_text(final_output, encoding="utf-8")
        (session_path / "session_meta.json").write_text(
            json.dumps(
                {
                    "id": session_path.name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return session_path / "final.txt"

    def list_sessions(self) -> list[SessionSummary]:
        sessions: list[SessionSummary] = []
        for path in sorted(self.root.glob("session_*"), reverse=True):
            if not path.is_dir():
                continue

            meta_path = path / "session_meta.json"
            mode = "unknown"
            created_at = path.name.replace("session_", "")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    mode = meta.get("mode", mode)
                    created_at = meta.get("created_at", created_at)
                except json.JSONDecodeError:
                    pass

            stages = sorted([p.name for p in path.glob("*_v*") if p.is_dir()])
            sessions.append(SessionSummary(id=path.name, created_at=created_at, mode=mode, stages=stages))

        return sessions

    def load_session(self, session_id: str) -> SessionDetails:
        session_path = self.root / session_id
        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        stages: dict[str, dict[str, Any]] = {}
        for stage_path in sorted([p for p in session_path.glob("*_v*") if p.is_dir()]):
            output_path = stage_path / "output.json"
            payload: dict[str, Any] = {}
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {"error": "Failed to decode output.json"}
            stages[stage_path.name] = payload

        final_output = ""
        final_path = session_path / "final.txt"
        if final_path.exists():
            final_output = final_path.read_text(encoding="utf-8")

        return SessionDetails(id=session_id, final_output=final_output, stages=stages)

    @staticmethod
    def _stable_hash(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
