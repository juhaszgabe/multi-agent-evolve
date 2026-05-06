"""
Entry point for the multi-agent data analysis system.

Usage:
    python src/main.py --csv <path> --question "<question>" [options]

Options:
    --output-dir DIR   Directory for charts (default: outputs)
    --log-dir DIR      Directory for structured logs (default: logs)
    --config PATH      Path to JSON config file (default: built-in defaults)

Router is selected by config.router_type:
  "planner"  — follows Planner's suggested_tool (default)
  "random"   — random valid action, seeded by config.random_seed
  "static"   — hand-coded mapping from config (requires config file)
  "bandit"   — not yet implemented
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _build_router(config):
    from router import PlannerRouter, RandomRouter, StaticRouter, BanditRouter, RouterAction

    if config.router_type == "planner":
        return PlannerRouter()
    elif config.router_type == "random":
        action_space = [
            RouterAction(a["agent_role"], a["tool_name"])
            for a in config.action_space
        ]
        return RandomRouter(action_space, seed=config.random_seed)
    elif config.router_type == "static":
        default_action = RouterAction("data_analysis", "python_sandbox")
        return StaticRouter({"default": default_action})
    elif config.router_type == "bandit":
        return BanditRouter()
    else:
        raise ValueError(f"Unknown router_type: {config.router_type!r}")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent CSV data analysis")
    parser.add_argument("--csv", required=True, help="Path to CSV or Excel file")
    parser.add_argument("--question", required=True, help="Natural-language question about the data")
    parser.add_argument("--output-dir", default="outputs", help="Directory for charts")
    parser.add_argument("--log-dir", default="logs", help="Directory for structured logs")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    from config import load_config
    from ai_providers.nvidia_provider import NvidiaProvider
    from orchestrator import run

    config = load_config(args.config)

    # Seed global RNG for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    provider = NvidiaProvider()
    router = _build_router(config)

    print(f"\nQuestion   : {args.question}")
    print(f"CSV        : {args.csv}")
    print(f"Router     : {config.router_type}")
    print(f"Temperature: {config.temperature}")
    print(f"Seed       : {config.random_seed}\n")

    state = run(
        question=args.question,
        csv_path=args.csv,
        provider=provider,
        config=config,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        router=router,
    )

    report = state.get("final_report") or {}
    print("\n" + "=" * 60)
    print("REPORT\n")
    print(report.get("report_text", "(no report text)"))

    findings = report.get("key_findings", [])
    if findings:
        print("\nKey findings:")
        for f in findings:
            print(f"  • {f}")

    caveats = report.get("caveats", [])
    if caveats:
        print("\nCaveats:")
        for c in caveats:
            print(f"  • {c}")

    charts = state.get("chart_paths", [])
    if charts:
        print(f"\nCharts: {charts}")

    # Save full trace for inspection / future eval pipeline
    os.makedirs(args.output_dir, exist_ok=True)
    trace_path = os.path.join(args.output_dir, f"trace_{state.get('workflow_id', 'unknown')}.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in state.items() if k not in ("schema", "step_records")},
            f, indent=2, default=str,
        )
    print(f"\nTrace : {trace_path}")
    print(f"Logs  : {args.log_dir}/step_records/{state.get('workflow_id', '')}.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
