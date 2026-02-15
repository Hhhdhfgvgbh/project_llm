from pathlib import Path

from app.core.output_cleaner import OutputCleaner


def test_output_cleaner_removes_universal_blocks(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """
version: 1
universal:
  remove_blocks:
    - start: "<think>"
      end: "</think>"
""",
        encoding="utf-8",
    )

    cleaner = OutputCleaner.from_yaml(rules)
    cleaned = cleaner.clean("<think>hidden</think>Visible answer", model_name="phi")
    assert cleaned == "Visible answer"


def test_output_cleaner_supports_model_specific_rules(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """
version: 1
universal:
  remove_blocks:
    - start: "<think>"
      end: "</think>"
model_rules:
  deepseek:
    remove_blocks:
      - start: "<thought>"
        end: "</thought>"
""",
        encoding="utf-8",
    )

    cleaner = OutputCleaner.from_yaml(rules)
    cleaned = cleaner.clean("<thought>secret</thought>Result", model_name="deepseek")
    assert cleaned == "Result"
