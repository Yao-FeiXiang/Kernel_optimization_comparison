"""Parse native benchmark records and maintain the generated result table."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BEGIN_MARKER = "<!-- SGEMM_RESULTS_BEGIN -->"
END_MARKER = "<!-- SGEMM_RESULTS_END -->"
_RECORD_PREFIX = "<!-- record:"
_RECORD_SUFFIX = " -->"


@dataclass(frozen=True)
class BenchmarkRecord:
    schema_version: int
    method_id: int
    method: str
    status: str
    m: int
    n: int
    k: int
    alpha: float
    beta: float
    warmup: int
    repeat: int
    latency_ms: float
    min_latency_ms: float
    gflops: float
    max_abs_error: float
    max_rel_error: float
    gpu: str
    compute_capability: str
    cuda_runtime: str
    timestamp_utc: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BenchmarkRecord":
        required = tuple(cls.__dataclass_fields__)
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"missing benchmark field(s): {', '.join(missing)}")
        selected = {name: values[name] for name in required}
        return cls(**selected)

    @property
    def experiment_key(self) -> tuple[Any, ...]:
        return (
            self.gpu,
            self.compute_capability,
            self.cuda_runtime,
            self.m,
            self.n,
            self.k,
            self.alpha,
            self.beta,
            self.warmup,
            self.repeat,
        )

    @property
    def row_key(self) -> tuple[Any, ...]:
        return (*self.experiment_key, self.method_id)


def upsert_records(
    existing: Sequence[BenchmarkRecord], new: Sequence[BenchmarkRecord]
) -> list[BenchmarkRecord]:
    indexed = {record.row_key: record for record in existing}
    indexed.update({record.row_key: record for record in new})
    return sorted(indexed.values(), key=lambda record: (record.experiment_key, record.method_id))


def _compact_json(record: BenchmarkRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _render_group(records: Sequence[BenchmarkRecord]) -> str:
    first = records[0]
    naive = next(
        (record for record in records if record.method_id == 1 and record.status == "pass"),
        None,
    )
    lines = [
        (
            f"## {first.gpu} — M={first.m}, N={first.n}, K={first.k}"
        ),
        "",
        (
            f"- Compute capability: `{first.compute_capability}`; CUDA runtime: "
            f"`{first.cuda_runtime}`; alpha={first.alpha:g}; beta={first.beta:g}"
        ),
        f"- Warmup: {first.warmup}; timed samples: {first.repeat}; updated: `{first.timestamp_utc}`",
        "- Reproduce: "
        f"`python3 benchmark.py --all --m {first.m} --n {first.n} --k {first.k} "
        "--update-results`",
        "",
    ]
    lines.extend(f"{_RECORD_PREFIX}{_compact_json(record)}{_RECORD_SUFFIX}" for record in records)
    lines.extend(
        [
            "",
            "| ID | Method | Status | Median ms | Min ms | GFLOP/s | vs Naive | Max abs error | Max rel error |",
            "|---:|:---|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        speedup = "—"
        if naive is not None and naive.gflops > 0 and record.status == "pass":
            speedup = f"{record.gflops / naive.gflops:.3f}x"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.method_id),
                    record.method,
                    record.status,
                    _format_number(record.latency_ms),
                    _format_number(record.min_latency_ms),
                    _format_number(record.gflops),
                    speedup,
                    f"{record.max_abs_error:.3e}",
                    f"{record.max_rel_error:.3e}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_results(records: Sequence[BenchmarkRecord]) -> str:
    groups: dict[tuple[Any, ...], list[BenchmarkRecord]] = {}
    for record in sorted(records, key=lambda item: (item.experiment_key, item.method_id)):
        groups.setdefault(record.experiment_key, []).append(record)
    if not groups:
        return "尚无本机测试结果。运行 `python3 benchmark.py --all --update-results` 生成。"
    return "\n\n".join(_render_group(group) for group in groups.values())


def _managed_region(text: str) -> str | None:
    if BEGIN_MARKER not in text or END_MARKER not in text:
        return None
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("results file must contain exactly one managed marker pair")
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER, start)
    return text[start:end]


def parse_managed_records(text: str) -> list[BenchmarkRecord]:
    region = _managed_region(text)
    if region is None:
        return []
    records: list[BenchmarkRecord] = []
    for raw_line in region.splitlines():
        line = raw_line.strip()
        if not (line.startswith(_RECORD_PREFIX) and line.endswith(_RECORD_SUFFIX)):
            continue
        payload = line[len(_RECORD_PREFIX) : -len(_RECORD_SUFFIX)]
        try:
            values = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid embedded benchmark record: {error}") from error
        records.append(BenchmarkRecord.from_mapping(values))
    return records


def _split_managed_region(text: str) -> tuple[str, str]:
    region = _managed_region(text)
    if region is None:
        prefix = text
        if not prefix:
            prefix = "# SGEMM Benchmark Results\n\n"
        elif not prefix.endswith("\n\n"):
            prefix = prefix.rstrip("\n") + "\n\n"
        return prefix, "\n"
    del region
    start = text.index(BEGIN_MARKER)
    end = text.index(END_MARKER, start) + len(END_MARKER)
    return text[:start], text[end:]


def update_results_file(path: Path, new_records: Iterable[BenchmarkRecord]) -> None:
    old_text = path.read_text(encoding="utf-8") if path.exists() else ""
    existing = parse_managed_records(old_text)
    records = upsert_records(existing, list(new_records))
    prefix, suffix = _split_managed_region(old_text)
    content = prefix + BEGIN_MARKER + "\n" + render_results(records) + "\n" + END_MARKER + suffix
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
