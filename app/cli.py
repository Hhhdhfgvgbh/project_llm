from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config.loader import ConfigLoader
from app.core.executors import BasePipelineExecutor, ManualPipelineExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="project_llm pipeline runner")
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline.yaml")
    parser.add_argument("--mode", choices=["base", "manual"], default="base")
    parser.add_argument("--input", required=True, help="User input text")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.mode == "base":
        result = BasePipelineExecutor(args.models, args.pipeline).run(args.input)
    else:
        pipeline = ConfigLoader.load_pipeline_config(Path(args.pipeline))
        result = ManualPipelineExecutor(args.models, pipeline).run(args.input)

    if args.json:
        print(
            json.dumps(
                {
                    "final_output": result.final_output,
                    "steps": [
                        {
                            "stage_id": step.stage_id,
                            "stage_type": step.stage_type,
                            "models": list(step.model_outputs.keys()),
                        }
                        for step in result.steps
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(result.final_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
