# Process

Flat rules. Deviations need Maxime's word, not good intentions.

## Testing
- **`dspy-ci` is the only test authority.** Never local pytest as the verdict. It snapshots the working tree (staged + unstaged) and accepts pytest args for focused runs; no args = full matrix (py3.10–3.14 + live-LM smoke on the 48-core server).
- **Stage new files before running it** — the snapshot excludes untracked files (`git stash create` semantics). This has burned two agents.
- Full matrix green before an epic is called done. The pre-push hook runs the matrix again; a one-off flake → rerun before investigating (known: `test_dspy_configure_allowance_async` py3.14).

## The corpus (L8, operational form)
- Fixtures regenerate only via `tests/adapters/golden/generate_fixtures.py` in a dedicated commit with zero `dspy/` source changes.
- Fixtures for a planned refactor land *before* it, pinning the behavior.
- Python-version-sensitive fixtures get the `python_sensitive` tag; the pinned parity CI job is the byte authority.

## Git
- Stacked commits in logical units; messages match repo history style (`refactor(adapters): …`, `feat(signatures): …`, `docs(programir): …`).
- **No AI attribution anywhere** — no Co-Authored-By, no "Generated with", in commits, PR bodies, or issues.
- **Never push** without Maxime's word. **Never open PRs** — Maxime says when.
- **Never touch stanfordnlp/dspy** (upstream isolation). Work happens on the MaximeRivest fork.
- `scratch/` is a symlink — nothing under it can be committed; durable docs go in `roadmap/`.
- Known noise: `uv run` drifts `uv.lock` — restore, don't commit; `review_packet/` is untracked scratch — leave it.

## Code
- Docstrings: Google format, Wickham voice — direct, concise, beginner-empathetic.
- Comments state constraints code can't show; never provenance, narration, or reviewer-reassurance.
- Sync and async paths change identically in every PR that touches either.
- Zero public-surface change unless the epic doc explicitly sanctions it (and the report must flag it).

## Epics
- Doc first, in `roadmap/epic-<X>-<name>.md`, quiet-compiler style: PR stack, mechanical definitions of done, corpus gates, non-goals. The implementing engineer writes it (the builder catches what the planner doesn't).
- End-of-epic: update `02-state.md` and append ratified decisions to `05-decisions.md`.
- During review sessions: restart difit (localhost:4966) after every commit.

## Review of upstream PRs
Run the five-point orthogonality check before sync: (1) no multi-call policy in adapters, (2) no new semantic types / provider hacks, (3) no rendering/parsing changes (corpus impact), (4) no undeclared prediction fields, (5) observability-only changes merge cleanly.
