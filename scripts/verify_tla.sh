#!/usr/bin/env bash
# Parse and model check the TLA+ specification rhoc generates for a program.
#
#   scripts/verify_tla.sh [program.rho]
#
# Needs a JRE and tla2tools.jar. Point TLA2TOOLS at the jar, or drop it in the
# repository root; releases live at https://github.com/tlaplus/tlaplus/releases.
#
# Programs that divide or use fractional literals produce a spec over Reals,
# which SANY parses but TLC cannot evaluate. This script says so and stops
# after parsing rather than reporting a failure.
set -euo pipefail

RHO="${1:-examples/gradient_2d.rho}"
JAR="${TLA2TOOLS:-tla2tools.jar}"

if ! command -v java >/dev/null 2>&1; then
  echo "java not found; install a JRE to model check." >&2
  exit 127
fi
if [ ! -f "$JAR" ]; then
  echo "tla2tools.jar not found at '$JAR'." >&2
  echo "Set TLA2TOOLS=/path/to/tla2tools.jar" >&2
  exit 127
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> compiling $RHO"
cargo run --quiet --release --bin rhoc -- "$RHO" -o "$WORK/kernel.so" --dump-tla >/dev/null
mv rho_harmony.tla rho_harmony.cfg "$WORK/"

echo "==> SANY (syntax and semantics)"
( cd "$WORK" && java -cp "$JAR" tla2sany.SANY rho_harmony.tla )

if grep -q "EXTENDS Reals" "$WORK/rho_harmony.tla"; then
  echo
  echo "==> spec is over Reals; TLC cannot evaluate real division."
  echo "    Parsed successfully. Skipping model checking."
  exit 0
fi

echo
echo "==> TLC (model checking)"
( cd "$WORK" && java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -nowarning rho_harmony.tla )
