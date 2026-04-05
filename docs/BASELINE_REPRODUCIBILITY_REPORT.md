# Baseline Reproducibility Report

Date: 2026-04-05

## Goal

Provide evaluator-ready evidence for baseline reproducibility with:

- per-task score spread
- runtime envelope
- exact run configuration and reproduction steps

## Run Configuration

| Field | Value |
|---|---|
| Repository commit | fe2435ccdf3cd07b8778c6442bb0e1b5aae8d48d |
| Baseline policy | heuristic |
| Split | eval |
| Seeds | 1, 2, 3, 4, 5, 6 |
| Python | 3.13.4 |
| Environment API | OpenEnv-compatible local env loop |

## Per-Task Score Spread

Heuristic baseline on eval split, 6 seeds:

| Task | Mean | Std | Min | Max | N |
|---|---:|---:|---:|---:|---:|
| 1 | 0.7850 | 0.0280 | 0.7350 | 0.8100 | 6 |
| 2 | 0.7975 | 0.0976 | 0.5850 | 0.8600 | 6 |
| 3 | 0.5338 | 0.3064 | 0.2769 | 0.9644 | 6 |
| 4 | 0.8376 | 0.0864 | 0.6444 | 0.8762 | 6 |
| 5 | 0.7389 | 0.2198 | 0.5191 | 0.9588 | 6 |

Overall mean score across seeds: 0.7386

## Runtime Envelope

Runtime below is measured as full 5-task baseline run time per seed (`runtime_seconds` from `run_baseline`).

| Metric | Seconds |
|---|---:|
| min | 2.10 |
| median | 3.16 |
| p95 | 4.20 |
| max | 4.52 |
| mean | 3.25 |

## Reproduction Steps

1. Run the same baseline configuration used for this report:

```bash
python3 -m inference \
  --policy-mode heuristic \
  --split eval \
  --seeds 1 2 3 4 5 6 \
  --stdout-protocol json > /tmp/mario_repro_eval_heuristic.json
```

2. Summarize task spread and runtime envelope:

```bash
python3 - <<'PY'
import json
from statistics import mean, median, pstdev

payload = json.load(open('/tmp/mario_repro_eval_heuristic.json', encoding='utf-8'))
runs = payload['runs']

runtime = [float(run['runtime_seconds']) for run in runs]
runtime_sorted = sorted(runtime)
p95 = runtime_sorted[max(0, int(0.95 * len(runtime_sorted)) - 1)] if runtime_sorted else 0.0

print('runtime_envelope_seconds=', {
    'min': round(min(runtime), 2),
    'median': round(median(runtime), 2),
    'p95': round(p95, 2),
    'max': round(max(runtime), 2),
    'mean': round(mean(runtime), 2),
})

for task_id in [1, 2, 3, 4, 5]:
    scores = [
        float(next(item['score'] for item in run['results'] if int(item['task_id']) == task_id))
        for run in runs
    ]
    print(
        f'task_{task_id}:',
        {
            'mean': round(mean(scores), 4),
            'std': round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
            'min': round(min(scores), 4),
            'max': round(max(scores), 4),
            'n': len(scores),
        },
    )
PY
```

## Reviewer Notes

- This report focuses on reproducibility requirements only (score spread and runtime envelope).
- Broader benchmark behavior and adaptation evidence are documented in:
  - `docs/REWARD_STRUCTURE_AND_ADAPTATION.md`
  - `docs/assets/benchmark_runs.json`
  - `docs/assets/adaptation_report.json`
