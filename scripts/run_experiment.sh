#!/usr/bin/env bash
# Run a named experiment preset with optional extra Hydra overrides.
#   ./scripts/run_experiment.sh example seed.value=7
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <experiment-preset> [hydra overrides...]" >&2
    exit 2
fi

preset="$1"
shift
cd "$(dirname "$0")/.."
exec uv run ips-r run "experiment=${preset}" "$@"
