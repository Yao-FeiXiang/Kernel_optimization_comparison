# Automatic Results Visual Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically render every `--update-results` experiment group as a GFLOP/s-ranked Markdown table with a visual performance bar relative to cuBLAS.

**Architecture:** Keep embedded JSON records and upsert keys unchanged. Add presentation-only ordering and bar formatting inside `sgemm_tools.results._render_group()`, so every existing update path regenerates the visualization without a new CLI option.

**Tech Stack:** Python 3 standard library, `unittest`, Markdown, existing `benchmark.py` result pipeline.

---

### Task 1: Specify ranking and bar behavior with failing tests

**Files:**
- Modify: `tests/test_results.py`

- [ ] **Step 1: Add failing tests for ranked rows and the cuBLAS bar**

Add tests that build real `BenchmarkRecord` objects and inspect the rendered Markdown:

```python
def test_render_ranks_methods_and_draws_performance_bar(self):
    naive = BenchmarkRecord.from_mapping(SAMPLE)
    tiled = BenchmarkRecord.from_mapping(
        {**SAMPLE, "method_id": 10, "method": "warptiling", "gflops": 5.0}
    )
    cublas = BenchmarkRecord.from_mapping(
        {**SAMPLE, "method_id": 0, "method": "cublas", "gflops": 10.0}
    )
    rendered = render_results([naive, tiled, cublas])
    rows = [line for line in rendered.splitlines() if line.startswith("| ")][1:]

    self.assertIn("| Rank |", rendered)
    self.assertTrue(rows[0].startswith("| 1 | 0 | cublas |"))
    self.assertTrue(rows[1].startswith("| 2 | 10 | warptiling |"))
    self.assertTrue(rows[2].startswith("| 3 | 1 | naive |"))
    self.assertIn("████████████████████ 100.0%", rows[0])
    self.assertIn("██████████ 50.0%", rows[1])

def test_render_omits_performance_bar_without_cublas(self):
    rendered = render_results([BenchmarkRecord.from_mapping(SAMPLE)])
    row = next(line for line in rendered.splitlines() if line.startswith("| 1 | 1 |"))
    self.assertIn("| — |", row)
```

Update the existing ordering assertion to expect cuBLAS before warptiling under performance ranking:

```python
self.assertLess(rendered.index("| 1 | 0 | cublas"), rendered.index("| 2 | 10 | warptiling"))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m unittest tests.test_results.BenchmarkRecordTest -v`

Expected: FAIL because the current table has no `Rank` column, retains method-ID order, and emits only a numeric `% cuBLAS` cell.

### Task 2: Implement visual performance ranking

**Files:**
- Modify: `sgemm_tools/results.py`
- Test: `tests/test_results.py`

- [ ] **Step 1: Add presentation ordering and bar helpers**

Add these focused helpers next to `_format_number()`:

```python
def _performance_order(record: BenchmarkRecord) -> tuple[Any, ...]:
    if record.status == "pass":
        return (0, -record.gflops, _method_order(record))
    return (1, 0.0, _method_order(record))


def _performance_bar(record: BenchmarkRecord, cublas: BenchmarkRecord | None) -> str:
    if record.status != "pass" or cublas is None or cublas.gflops <= 0:
        return "—"
    percent = 100.0 * record.gflops / cublas.gflops
    block_count = min(20, max(0, int(percent / 5.0 + 0.5)))
    blocks = "█" * block_count
    return f"{blocks} {percent:.1f}%".lstrip()
```

- [ ] **Step 2: Render ranked rows with a visual column**

Inside `_render_group()`, keep embedded record comments in their existing order, then create:

```python
display_records = sorted(records, key=_performance_order)
rank_by_key = {
    record.row_key: rank
    for rank, record in enumerate(
        (item for item in display_records if item.status == "pass"), start=1
    )
}
```

Change the header to include `Rank` and `Performance vs cuBLAS`, iterate over `display_records`, place `rank_by_key.get(record.row_key, "—")` first, and use `_performance_bar(record, cublas)` in the visual column. Retain median, minimum, GFLOP/s, `vs Naive`, and error cells.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run: `python3 -m unittest tests.test_results.BenchmarkRecordTest -v`

Expected: all `BenchmarkRecordTest` tests PASS.

- [ ] **Step 4: Run the complete unit suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests PASS; CUDA integration tests may be skipped only when the driver or device is unavailable.

- [ ] **Step 5: Commit generator and tests**

```bash
git add -- sgemm_tools/results.py tests/test_results.py
git commit -m "feat: visualize benchmark performance rankings"
```

### Task 3: Document and regenerate current results

**Files:**
- Modify: `README.md`
- Modify: `RESULTS.md`
- Test: `tests/test_docs.py`

- [ ] **Step 1: Add a failing documentation assertion**

Extend the README contract test with:

```python
self.assertIn("Performance vs cuBLAS", self.readme)
self.assertIn("GFLOP/s", self.readme)
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `python3 -m unittest tests.test_docs.DocumentationTest -v`

Expected: FAIL because README does not yet describe the visual ranking.

- [ ] **Step 3: Document automatic visualization**

After the result update explanation in `README.md`, state that each experiment group is sorted by GFLOP/s and that `Performance vs cuBLAS` uses a 20-block bar; groups without a successful cuBLAS record show `—`.

- [ ] **Step 4: Run documentation tests and verify GREEN**

Run: `python3 -m unittest tests.test_docs.DocumentationTest -v`

Expected: all documentation tests PASS.

- [ ] **Step 5: Re-render the existing experiment records without rerunning kernels**

Run:

```bash
python3 -c 'from pathlib import Path; from sgemm_tools.results import update_results_file; update_results_file(Path("RESULTS.md"), [])'
```

Expected: the existing A100 records remain intact and the managed region displays the new ranked visual table.

- [ ] **Step 6: Verify generated content and repository checks**

Run:

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Expected: all tests PASS and `git diff --check` exits 0.

- [ ] **Step 7: Commit documentation and generated results**

```bash
git add -- README.md RESULTS.md tests/test_docs.py
git commit -m "docs: refresh visual benchmark results"
```
