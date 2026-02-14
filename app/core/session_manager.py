from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, UTC
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


class SessionManager:
    def __init__(self, root: Path | str = "sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> Path:
        session_id = datetime.now(UTC).strftime("session_%Y%m%d_%H%M%S")
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
            timestamp=datetime.now(UTC).isoformat(),
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

    @staticmethod
    def _stable_hash(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
