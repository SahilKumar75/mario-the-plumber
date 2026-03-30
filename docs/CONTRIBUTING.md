# Contributing

This repo is organized around the `pipeline_doctor/` submission project.

## Where To Work

- Core environment code lives in `../pipeline_doctor/`
- The latest working spec lives in `CONTEXT 3.md`
- Submission validation lives in `../pipeline_doctor/scripts/validate-submission.sh`

## Recommended GitHub Workflow

To keep the contribution graph more diverse and make history easier to read:

1. Open an issue for each meaningful task.
2. Create a feature branch from `main`.
3. Make focused changes and commit them with a clear message.
4. Push the branch and open a pull request.
5. Merge the PR instead of stacking all work directly on `main`.

Suggested branch prefix:

- `codex/issue-<number>-short-topic`

## Submission Checklist

Before shipping a new submission-oriented change:

1. Run `python3 inference.py` inside `../pipeline_doctor/`
2. Run `openenv validate` inside `../pipeline_doctor/`
3. If deployed, run `../pipeline_doctor/scripts/validate-submission.sh <space-url> ../pipeline_doctor`
4. Confirm the required env vars are documented:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`

## Notes

- Keep the root repo lightweight and use it mainly to organize the Mario workspace
- Put environment-specific code and docs under `pipeline_doctor/`
