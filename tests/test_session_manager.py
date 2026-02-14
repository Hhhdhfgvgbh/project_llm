from pathlib import Path

from app.core.session_manager import SessionManager


def test_session_manager_writes_versioned_stage(tmp_path: Path) -> None:
    manager = SessionManager(root=tmp_path)
    session = manager.create_session()

    stage_dir = manager.write_stage_result(
        session_path=session,
        stage_id="stage2",
        version=2,
        parent_version=1,
        input_payload={"q": "hello"},
        config_payload={"temperature": 0.7},
        output_payload={"answer": "world"},
    )

    manager.write_final_output(session, "final text", mode="base")

    assert stage_dir.name == "stage2_v2"
    assert (stage_dir / "meta.json").exists()
    assert (stage_dir / "input.json").exists()
    assert (stage_dir / "config.json").exists()
    assert (stage_dir / "output.json").exists()

    sessions = manager.list_sessions()
    assert len(sessions) == 1
    loaded = manager.load_session(sessions[0].id)
    assert loaded.final_output == "final text"
    assert "stage2_v2" in loaded.stages
