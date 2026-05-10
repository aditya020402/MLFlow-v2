"""CLI entry point for ada-ds.

Usage:
    ada run --dataset data.csv --instructions instructions.md [--output ./ada_output] [--env .env]
"""

import argparse
import sys


def _cmd_run(args: argparse.Namespace) -> None:
    from ada.run import run_pipeline
    try:
        result = run_pipeline(
            dataset=args.dataset,
            instructions=args.instructions,
            output_dir=args.output,
            env_file=args.env,
            verbose=not args.quiet,
        )
        if result["status"] != "completed":
            print(f"[ADA] Pipeline ended with status: {result['status']}", file=sys.stderr)
            sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ADA] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ADA] Interrupted.", file=sys.stderr)
        sys.exit(130)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="Ada — The Digital Data Scientist. Run an end-to-end ML pipeline locally.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the ML pipeline on a dataset.")
    run_parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to the input CSV dataset.",
    )
    run_parser.add_argument(
        "--instructions", "-i", required=True,
        help="Path to the .md instructions file (task type, target column, etc.).",
    )
    run_parser.add_argument(
        "--output", "-o", default="./ada_output",
        help="Output directory for models, plots, and reports. (default: ./ada_output)",
    )
    run_parser.add_argument(
        "--env", "-e", default=".env",
        help="Path to .env file containing OPENAI_API_KEY. (default: .env)",
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress INFO logging.",
    )

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
