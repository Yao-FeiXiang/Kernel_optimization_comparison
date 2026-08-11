import json
import tempfile
import unittest
from pathlib import Path

from sgemm_tools.results import (
    BEGIN_MARKER,
    END_MARKER,
    BenchmarkRecord,
    parse_managed_records,
    render_results,
    update_results_file,
    upsert_records,
)


SAMPLE = {
    "schema_version": 1,
    "method_id": 1,
    "method": "naive",
    "status": "pass",
    "m": 64,
    "n": 65,
    "k": 33,
    "alpha": 0.8,
    "beta": 0.2,
    "warmup": 2,
    "repeat": 5,
    "latency_ms": 0.125,
    "min_latency_ms": 0.120,
    "gflops": 2.21,
    "max_abs_error": 0.0001,
    "max_rel_error": 0.00001,
    "gpu": "Test GPU",
    "compute_capability": "8.6",
    "cuda_runtime": "13.0",
    "timestamp_utc": "2026-08-11T00:00:00Z",
}


class BenchmarkRecordTest(unittest.TestCase):
    def test_phase_b_fields_are_optional_for_old_records(self):
        record = BenchmarkRecord.from_mapping(SAMPLE)
        self.assertTrue(record.available)
        self.assertEqual(record.implementation_variant, "")
        self.assertEqual(record.configuration, "")

    def test_phase_b_fields_are_preserved_when_present(self):
        record = BenchmarkRecord.from_mapping(
            {
                **SAMPLE,
                "available": False,
                "implementation_variant": "scalar-fallback",
                "configuration": "BM64_BN64_BK8_TM8_TN8",
            }
        )
        self.assertFalse(record.available)
        self.assertEqual(record.implementation_variant, "scalar-fallback")
        self.assertEqual(record.configuration, "BM64_BN64_BK8_TM8_TN8")

    def test_from_mapping_accepts_unknown_fields(self):
        record = BenchmarkRecord.from_mapping({**SAMPLE, "future_field": 123})
        self.assertEqual(record.method, "naive")
        self.assertEqual(record.m, 64)

    def test_from_mapping_rejects_missing_required_field(self):
        broken = dict(SAMPLE)
        del broken["method"]
        with self.assertRaisesRegex(ValueError, "method"):
            BenchmarkRecord.from_mapping(broken)

    def test_upsert_replaces_same_configuration_and_method(self):
        first = BenchmarkRecord.from_mapping(SAMPLE)
        faster = BenchmarkRecord.from_mapping({**SAMPLE, "gflops": 3.0})
        records = upsert_records([first], [faster])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].gflops, 3.0)

    def test_upsert_preserves_a_different_matrix_size(self):
        first = BenchmarkRecord.from_mapping(SAMPLE)
        other = BenchmarkRecord.from_mapping({**SAMPLE, "m": 128})
        records = upsert_records([first], [other])
        self.assertEqual({record.m for record in records}, {64, 128})

    def test_rendered_records_round_trip_without_rounded_value_loss(self):
        record = BenchmarkRecord.from_mapping({**SAMPLE, "gflops": 2.211234567})
        rendered = render_results([record])
        parsed = parse_managed_records(
            f"# Results\n\n{BEGIN_MARKER}\n{rendered}\n{END_MARKER}\n"
        )
        self.assertEqual(parsed, [record])
        self.assertIn("| 1 | naive | pass |", rendered)

    def test_render_calculates_speedup_against_naive_in_same_experiment(self):
        naive = BenchmarkRecord.from_mapping(SAMPLE)
        tiled = BenchmarkRecord.from_mapping(
            {**SAMPLE, "method_id": 5, "method": "blocktiling-2d", "gflops": 4.42}
        )
        rendered = render_results([naive, tiled])
        self.assertIn("| 5 | blocktiling-2d | pass |", rendered)
        self.assertIn("| 2.000x |", rendered)

    def test_render_shows_variant_configuration_and_percent_cublas(self):
        naive = BenchmarkRecord.from_mapping(SAMPLE)
        tiled = BenchmarkRecord.from_mapping(
            {
                **SAMPLE,
                "method_id": 10,
                "method": "warptiling",
                "gflops": 8.84,
                "implementation_variant": "warp-tiled",
                "configuration": "BM128_BN128_BK16",
            }
        )
        cublas = BenchmarkRecord.from_mapping(
            {**SAMPLE, "method_id": 0, "method": "cublas", "gflops": 10.0}
        )
        rendered = render_results([cublas, naive, tiled])
        self.assertIn("Variant / configuration", rendered)
        self.assertIn("warp-tiled / BM128_BN128_BK16", rendered)
        self.assertIn("| 88.400% |", rendered)
        self.assertLess(rendered.index("| 10 | warptiling"), rendered.index("| 0 | cublas"))

    def test_unavailable_method_renders_without_numeric_metrics(self):
        unavailable = BenchmarkRecord.from_mapping(
            {
                **SAMPLE,
                "method_id": 0,
                "method": "cublas",
                "status": "unavailable",
                "available": False,
                "latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "gflops": 0.0,
            }
        )
        rendered = render_results([BenchmarkRecord.from_mapping(SAMPLE), unavailable])
        row = next(line for line in rendered.splitlines() if line.startswith("| 0 |"))
        self.assertIn("| unavailable | — | — | — | — | — |", row)


class ResultsFileTest(unittest.TestCase):
    def test_update_is_idempotent_and_preserves_text_outside_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULTS.md"
            path.write_text(
                f"# Local heading\n\n{BEGIN_MARKER}\n{END_MARKER}\n\nFooter\n",
                encoding="utf-8",
            )
            record = BenchmarkRecord.from_mapping(SAMPLE)

            update_results_file(path, [record])
            once = path.read_text(encoding="utf-8")
            update_results_file(path, [record])
            twice = path.read_text(encoding="utf-8")

            self.assertEqual(once, twice)
            self.assertTrue(twice.startswith("# Local heading"))
            self.assertTrue(twice.endswith("Footer\n"))
            self.assertEqual(twice.count(BEGIN_MARKER), 1)
            self.assertEqual(twice.count(END_MARKER), 1)
            self.assertFalse(path.with_suffix(".md.tmp").exists())

    def test_update_creates_a_new_results_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULTS.md"
            update_results_file(path, [BenchmarkRecord.from_mapping(SAMPLE)])
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# SGEMM Benchmark Results"))
            self.assertIn(BEGIN_MARKER, text)

    def test_embedded_record_is_valid_compact_json(self):
        record = BenchmarkRecord.from_mapping(SAMPLE)
        rendered = render_results([record])
        line = next(line for line in rendered.splitlines() if line.startswith("<!-- record:"))
        payload = line.removeprefix("<!-- record:").removesuffix(" -->")
        self.assertEqual(json.loads(payload)["method"], "naive")


if __name__ == "__main__":
    unittest.main()
