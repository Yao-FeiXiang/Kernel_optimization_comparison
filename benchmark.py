#!/usr/bin/env python3
"""Build, run, compare, and record the CUDA SGEMM teaching kernels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sgemm_tools.results import update_results_file
from sgemm_tools.runner import (
    BuildError,
    NativeRunOptions,
    ResultParseError,
    build_executable,
    parse_json_lines,
    run_native,
)


ROOT = Path(__file__).resolve().parent


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and benchmark the CUDA SGEMM optimization stages."
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--kernel", help="tutorial ID or stable method name")
    selectors.add_argument("--all", action="store_true", help="run all registered methods")
    selectors.add_argument("--tune", action="store_true", help="run the autotuned method")
    selectors.add_argument("--list", action="store_true", help="list methods without using a GPU")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--check", action="store_true", help="validate before timing")
    parser.add_argument("--check-only", action="store_true", help="validate without timing")
    build = parser.add_mutually_exclusive_group()
    build.add_argument("--build", action="store_true", help="force a native rebuild")
    build.add_argument("--no-build", action="store_true", help="require an existing executable")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--update-results", action="store_true")
    parser.add_argument("--results-file", type=Path, default=Path("RESULTS.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    build_dir = arguments.build_dir
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    executable = build_dir / "sgemm_bench"
    try:
        if arguments.no_build:
            if not executable.exists():
                raise BuildError(f"native executable does not exist: {executable}")
        else:
            executable = build_executable(ROOT, build_dir, force=arguments.build)

        if arguments.list:
            result = subprocess.run(
                [str(executable), "--list"], check=False, capture_output=True, text=True
            )
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode

        options = NativeRunOptions(
            kernel=arguments.kernel,
            all=arguments.all,
            tune=arguments.tune,
            m=arguments.m,
            n=arguments.n,
            k=arguments.k,
            alpha=arguments.alpha,
            beta=arguments.beta,
            warmup=arguments.warmup,
            repeat=arguments.repeat,
            seed=arguments.seed,
            atol=arguments.atol,
            rtol=arguments.rtol,
            check=arguments.check,
            check_only=arguments.check_only,
            json=True,
        )
        result = run_native(executable, options)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if arguments.update_results:
            records = parse_json_lines(result.stdout)
            if records:
                results_file = arguments.results_file
                if not results_file.is_absolute():
                    results_file = ROOT / results_file
                update_results_file(results_file, records)
                print(f"Updated {results_file}")
        return result.returncode
    except (BuildError, ResultParseError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
