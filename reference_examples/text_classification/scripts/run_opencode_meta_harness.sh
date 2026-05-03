#!/usr/bin/env bash
# Launch the form-filling Meta-Harness run with OpenCode as proposer and
# Together DeepSeek as the fixed solver.
#
# Usage:
#   TOGETHER_API_KEY=... scripts/run_opencode_meta_harness.sh [meta_harness.py args...]
#
# Common overrides:
#   RUN_NAME=my-run
#   ITERATIONS=1
#   PROPOSE_TIMEOUT=2400
#   SOLVER_MODEL=together_ai/deepseek-ai/DeepSeek-V4-Pro
#   OPENCODE_WRAPPER_MODEL=togetherai/deepseek-ai/DeepSeek-V4-Pro

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$EXAMPLE_DIR"

# Load local env if present. This file should not be committed.
if [[ -f ".env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source ".env"
    set +a
fi

SOLVER_MODEL="${SOLVER_MODEL:-together_ai/deepseek-ai/DeepSeek-V4-Pro}"
OPENCODE_MODEL_VALUE="${OPENCODE_WRAPPER_MODEL:-${OPENCODE_MODEL:-togetherai/deepseek-ai/DeepSeek-V4-Pro}}"
ITERATIONS="${ITERATIONS:-1}"
PROPOSE_TIMEOUT="${PROPOSE_TIMEOUT:-2400}"
RUN_NAME="${RUN_NAME:-}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required but was not found on PATH." >&2
    exit 1
fi

if ! command -v opencode >/dev/null 2>&1; then
    echo "ERROR: opencode is required but was not found on PATH." >&2
    echo "Install/configure OpenCode first, then rerun this script." >&2
    exit 1
fi

if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
    echo "ERROR: TOGETHER_API_KEY is not set." >&2
    echo "Set it in your shell or in reference_examples/text_classification/.env." >&2
    exit 1
fi

export META_HARNESS_PROPOSER=opencode
export OPENCODE_WRAPPER_MODEL="$OPENCODE_MODEL_VALUE"
export OPENCODE_MODEL="$OPENCODE_MODEL_VALUE"

echo "example_dir:       $EXAMPLE_DIR"
echo "proposer:          opencode"
echo "opencode_model:    $OPENCODE_WRAPPER_MODEL"
echo "solver_model:      $SOLVER_MODEL"
echo "iterations:        $ITERATIONS"
echo "propose_timeout:   $PROPOSE_TIMEOUT"
if [[ -n "$RUN_NAME" ]]; then
    echo "run_name:          $RUN_NAME"
fi
echo ""

cmd=(
    uv run python meta_harness.py
    --proposer opencode
    --model "$SOLVER_MODEL"
    --iterations "$ITERATIONS"
    --propose-timeout "$PROPOSE_TIMEOUT"
)

if [[ -n "$RUN_NAME" ]]; then
    cmd+=(--run-name "$RUN_NAME")
fi

if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
