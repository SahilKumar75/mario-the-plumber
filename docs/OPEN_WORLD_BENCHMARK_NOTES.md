# Open-World Benchmark Notes

This note captures the benchmark changes inspired by the latest RL, Gymnasium, open-world ML, and synthetic-data papers.

## What Changed

Mario now goes beyond fixed corruption templates in five concrete ways:

1. **Open-world failure patterns**
   - schema alias drift such as `event_time -> event_ts` and `product_segment -> category`
   - timestamp timezone drift and mixed datetime formatting
   - sentinel missing values such as `unknown` and `missing`
   - stale downstream summaries paired with delayed upstream recovery

2. **Clearer episode-budget semantics**
   - observations now expose `time_budget_remaining` and `time_budget_ratio`
   - episode endings distinguish commit success/failure from budget truncation
   - `done_reason` now records whether the run ended because of:
     - `commit_success`
     - `commit_failure`
     - `quality_collapse`
     - `step_budget_exhausted`

3. **Adaptive scenario profiles**
   - each task now samples from a family of scenario profiles
   - `train` and `eval` splits draw from different profile mixes
   - episodes expose:
     - `scenario_profile`
     - `open_world_patterns`
   - this makes the benchmark closer to an open-world evaluation than a single scripted scenario

4. **Stronger synthetic-data utility story**
   - benchmark metadata now explicitly documents how synthetic data is used
   - the utility claim is not “perfect realism”
   - the utility claim is that synthetic scenarios preserve **policy separation**
     - random policies stay near broken-state performance
     - structured policies recover the pipeline
     - held-out `eval` split still works

5. **Benchmark metadata and reporting**
   - `/benchmark-metadata` endpoint added
   - `scripts/benchmark_models.py` now exports JSON and CSV reports
   - `scripts/export_benchmark_metadata.py` exports scenario-profile coverage and initial-score statistics

## Why This Matters

The recent papers imply that serious benchmarks should:

- separate **termination** from **truncation**
- support **generalization** across varied scenario families
- expose enough metadata to reproduce and audit benchmark results
- treat synthetic data as useful only when it preserves meaningful algorithm comparisons

Mario now does each of these better than before.

## Synthetic Data Position

Mario uses synthetic data to create a **controlled evaluation space** for ETL repair and online recovery.

That means:

- schemas and dependencies are realistic enough to exercise agent behavior
- privacy-sensitive real data is not required
- scenario diversity can be expanded without leaking operational data

Known limitation:

- Mario still models a narrow slice of enterprise ETL relative to full production systems
- the utility story is benchmark-centric, not “drop-in substitute for real warehouse logs”

## New Commands

Export benchmark metadata:

```bash
python3 scripts/export_benchmark_metadata.py --seeds 1 2 3 4 5 --output docs/assets/benchmark_metadata.json
```

Export benchmark results:

```bash
python3 scripts/benchmark_models.py \
  --policies random heuristic hybrid \
  --splits train eval \
  --seeds 1 2 3 4 5 \
  --json-out docs/assets/benchmark_runs.json \
  --csv-out docs/assets/benchmark_runs.csv
```
