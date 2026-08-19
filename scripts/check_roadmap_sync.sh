#!/usr/bin/env bash
# Roadmap drift guard.
#
# The governance home for roadmap/ is the programir-main branch (worktree:
# ~/Projects/MaximeRivest-dspy). Rule enforced here:
#
#   - A roadmap/ file that EXISTS on programir-main must be byte-identical
#     on this branch. Editing the shared copy here is drift -> FAIL.
#   - A file that exists ONLY here is a pending-upstream draft -> OK,
#     listed for awareness.
#   - A file that exists only on programir-main is missing here -> FAIL
#     (copy it over with: git checkout programir-main -- roadmap/<file>).
#
# Ratification flow: draft here -> ratify on programir-main -> copy back.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UPSTREAM="programir-main"
if ! git rev-parse --verify -q "$UPSTREAM" >/dev/null; then
    echo "roadmap-sync: branch $UPSTREAM not found; skipping." >&2
    exit 0
fi

fail=0
pending=()

while IFS= read -r file; do
    if [ ! -f "$file" ]; then
        echo "roadmap-sync: MISSING here (exists on $UPSTREAM): $file" >&2
        echo "  fix: git checkout $UPSTREAM -- '$file'" >&2
        fail=1
    elif ! git diff --quiet "$UPSTREAM" -- "$file" 2>/dev/null \
            || ! cmp -s <(git show "$UPSTREAM:$file") "$file"; then
        echo "roadmap-sync: DRIFT from $UPSTREAM: $file" >&2
        echo "  shared files are edited on $UPSTREAM and copied back, never forked here." >&2
        fail=1
    fi
done < <(git ls-tree -r --name-only "$UPSTREAM" -- roadmap/)

while IFS= read -r file; do
    if ! git cat-file -e "$UPSTREAM:$file" 2>/dev/null; then
        pending+=("$file")
    fi
done < <(git ls-files -- roadmap/)

if [ "${#pending[@]}" -gt 0 ]; then
    echo "roadmap-sync: pending upstream (greenfield-only, OK):"
    printf '  %s\n' "${pending[@]}"
fi

exit "$fail"
