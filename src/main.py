"""
Entry point for the multi-agent data analysis system.

Usage:
    python src/main.py --csv <path> --question "<question>" [--output-dir outputs]

Environment variables (via .env):
    OPENAI_API_KEY   — NVIDIA NIM API key
    MODEL_SONNET     — model for Planner/DataAnalyst/Critic/Writer (default: meta/llama-3.3-70b-instruct)
    MODEL_HAIKU      — model for Visualizer (default: meta/llama-3.1-8b-instruct)
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Multi-agent CSV data analysis")
    parser.add_argument("--csv", required=True, help="Path to CSV or Excel file")
    parser.add_argument("--question", required=True, help="Natural-language question about the data")
    parser.add_argument("--output-dir", default="outputs", help="Directory for charts and reports")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    from ai_providers.nvidia_provider import NvidiaProvider
    from orchestrator import run

    provider = NvidiaProvider()
    model_sonnet = os.getenv("MODEL_SONNET", "meta/llama-3.3-70b-instruct")
    model_haiku = os.getenv("MODEL_HAIKU", "meta/llama-3.1-8b-instruct")

    print(f"\nQuestion : {args.question}")
    print(f"CSV      : {args.csv}")
    print(f"Models   : sonnet={model_sonnet}  haiku={model_haiku}\n")

    state = run(
        question=args.question,
        csv_path=args.csv,
        provider=provider,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        output_dir=args.output_dir,
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
        print(f"\nCharts saved: {charts}")

    # Save full state as JSON trace
    os.makedirs(args.output_dir, exist_ok=True)
    trace_path = os.path.join(args.output_dir, "trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in state.items() if k != "schema"},
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull trace: {trace_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
