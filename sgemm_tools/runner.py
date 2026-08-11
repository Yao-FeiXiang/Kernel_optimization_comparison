"""Build and execute the native SGEMM benchmark without shell interpolation."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .results import BenchmarkRecord


PHASE_A_SOURCES = (
    "src/benchmark.cu",
    "src/reference.cpp",
    "src/kernels/registry.cu",
    "src/kernels/kernel_01_naive.cu",
    "src/kernels/kernel_02_coalesced.cu",
    "src/kernels/kernel_03_shared.cu",
    "src/kernels/kernel_04_blocktiling_1d.cu",
    "src/kernels/kernel_05_blocktiling_2d.cu",
    "src/kernels/kernel_06_vectorized.cu",
)


class BuildError(RuntimeError):
    """Raised when the native benchmark cannot be built."""


class ResultParseError(ValueError):
    """Raised when a native line looks like JSON but is invalid."""


@dataclass(frozen=True)
class NativeRunOptions:
    kernel: Optional[str] = None
    all: bool = False
    m: int = 1024
    n: int = 1024
    k: int = 1024
    alpha: float = 0.8
    beta: float = 0.2
    warmup: int = 5
    repeat: int = 20
    seed: int = 42
    atol: Optional[float] = None
    rtol: Optional[float] = None
    check: bool = False
    check_only: bool = False
    json: bool = True


def native_arguments(options: NativeRunOptions) -> list[str]:
    if options.kernel is not None and options.all:
        raise ValueError("kernel and all are mutually exclusive")
    if options.kernel is not None:
        arguments = ["--kernel", options.kernel]
    elif options.all:
        arguments = ["--all"]
    else:
        raise ValueError("a kernel or all must be selected")
    arguments.extend(
        [
            "--m",
            str(options.m),
            "--n",
            str(options.n),
            "--k",
            str(options.k),
            "--alpha",
            str(options.alpha),
            "--beta",
            str(options.beta),
            "--warmup",
            str(options.warmup),
            "--repeat",
            str(options.repeat),
            "--seed",
            str(options.seed),
        ]
    )
    if options.atol is not None:
        arguments.extend(["--atol", str(options.atol)])
    if options.rtol is not None:
        arguments.extend(["--rtol", str(options.rtol)])
    if options.check_only:
        arguments.append("--check-only")
    elif options.check:
        arguments.append("--check")
    if options.json:
        arguments.append("--json")
    return arguments


def parse_json_lines(output: str) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResultParseError(f"invalid native JSON result: {error}") from error
        if not isinstance(values, dict):
            raise ResultParseError("native JSON result must be an object")
        try:
            records.append(BenchmarkRecord.from_mapping(values))
        except (TypeError, ValueError) as error:
            raise ResultParseError(str(error)) from error
    return records


def nvcc_command(root: Path, output: Path) -> list[str]:
    return [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        f"-I{root / 'include'}",
        *(str(root / source) for source in PHASE_A_SOURCES),
        "-o",
        str(output),
    ]


def _run_build(command: Sequence[str], root: Path) -> None:
    result = subprocess.run(
        list(command), cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise BuildError(f"build command failed ({result.returncode}): {' '.join(command)}\n{details}")


def build_executable(
    root: Path,
    build_dir: Path,
    *,
    force: bool = False,
    prefer_cmake: bool = True,
) -> Path:
    root = root.resolve()
    build_dir = build_dir.resolve()
    executable = build_dir / "sgemm_bench"
    if executable.exists() and not force:
        return executable
    build_dir.mkdir(parents=True, exist_ok=True)

    if prefer_cmake and shutil.which("cmake") is not None:
        configure = ["cmake", "-S", str(root), "-B", str(build_dir)]
        if shutil.which("ninja") is not None:
            configure.extend(["-G", "Ninja"])
        _run_build(configure, root)
        _run_build(["cmake", "--build", str(build_dir), "--target", "sgemm_bench"], root)
    else:
        if shutil.which("nvcc") is None:
            raise BuildError("neither CMake with CUDA nor nvcc is available")
        _run_build(nvcc_command(root, executable), root)
    if not executable.exists():
        raise BuildError(f"build succeeded but executable was not created: {executable}")
    return executable


def run_native(executable: Path, options: NativeRunOptions) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *native_arguments(options)],
        check=False,
        capture_output=True,
        text=True,
    )
