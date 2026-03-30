# Contributing

This repo itself is the Mario the Plumber submission project.

## Where To Work

- Core environment code lives at the repo root
- The latest working spec lives in `docs/CONTEXT 3.md`
- Submission validation lives in `scripts/validate-submission.sh`

## Recommended GitHub Workflow

To keep the contribution graph more diverse and make history easier to read:

1. Open an issue for each meaningful task.
2. Create a feature branch from `main`.
3. Make focused changes and commit them with a clear message.
4. Push the branch and open a pull request.
5. Merge the PR instead of stacking all work directly on `main`.

Suggested branch prefix:

- `issue-<number>-short-topic`

## Submission Checklist

Before shipping a new submission-oriented change:

1. Run `python3 inference.py` from the repo root
2. Run `openenv validate` from the repo root
3. If deployed, run `scripts/validate-submission.sh <space-url> .`
4. Confirm the required env vars are documented:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

## Notes

- Keep the repo root aligned with the actual submission layout
- Keep extra planning/history docs under `docs/`
