#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Check every TLA+ model in this directory with TLC (TXG-MODEL-005).
#
# Usage: TLA2TOOLS_JAR=/path/to/tla2tools.jar ./check.sh
#
# TLA2TOOLS_JAR names the official tla2tools.jar of TLA+ v1.7.4 (TLC2 2.19):
#   curl -sSfL -o tla2tools.jar \
#     https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar
# whose SHA-1 is bee4a54f3ee3d4afc347c3240ec2d9e93b075104. Keep the jar outside
# the repository. Each model X.tla is checked with its configuration X.cfg;
# TLC's state files go to a temporary directory that is removed on exit.
set -eu

: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to the official tla2tools.jar (v1.7.4)}"
[ -f "$TLA2TOOLS_JAR" ] || { echo "no such file: $TLA2TOOLS_JAR" >&2; exit 2; }

expected=bee4a54f3ee3d4afc347c3240ec2d9e93b075104
if command -v sha1sum >/dev/null 2>&1; then
    actual=$(sha1sum "$TLA2TOOLS_JAR" | cut -d' ' -f1)
elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 1 "$TLA2TOOLS_JAR" | cut -d' ' -f1)
else
    actual=unchecked
fi
if [ "$actual" != unchecked ] && [ "$actual" != "$expected" ]; then
    echo "$TLA2TOOLS_JAR has SHA-1 $actual, expected $expected" >&2
    exit 2
fi

cd "$(dirname "$0")"
meta=$(mktemp -d)
trap 'rm -rf "$meta"' EXIT INT TERM

status=0
for tla in *.tla; do
    name=${tla%.tla}
    log="$meta/$name.log"
    start=$(date +%s)
    if java -XX:+UseParallelGC -cp "$TLA2TOOLS_JAR" tlc2.TLC -workers 1 \
            -metadir "$meta/$name" -config "$name.cfg" "$tla" >"$log" 2>&1 \
        && grep -q "No error has been found" "$log"; then
        result=ok
    else
        result=FAILED
        status=1
    fi
    states=$(grep -E 'distinct states found, 0 states left' "$log" | tail -1 || true)
    printf '%-18s %-6s %3ss  %s\n' "$name" "$result" "$(( $(date +%s) - start ))" "$states"
    if [ "$result" = FAILED ]; then
        tail -40 "$log"
    fi
done
exit $status
