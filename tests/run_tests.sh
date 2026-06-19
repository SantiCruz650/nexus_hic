#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$HOME/venv/bin/python"
CI_REPORT="$DIR/ci_report.json"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       NEXUS HIC — CONTINUOUS INTEGRATION SUITE          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

overall_rc=0
results=()

run_test_block() {
    local name="$1"
    local script="$2"
    local report_var="$3"

    echo "────────────────────────────────────────────────────────"
    echo "  [CI] Block: $name"
    echo "  [CI] Script: $script"
    echo "────────────────────────────────────────────────────────"

    if $VENV_PYTHON "$script" 2>&1; then
        local rc=$?
    else
        local rc=$?
    fi

    if [ $rc -eq 0 ]; then
        echo "  [CI] $name: OK (exit code $rc)"
    else
        echo "  [CI] $name: FAIL (exit code $rc)"
    fi
    echo ""

    return $rc
}


echo "========================================================"
echo "  BLOCK 1/3 — Correctness (Precision Matematica)"
echo "========================================================"
if run_test_block "Correctness" "$DIR/test_correctness.py"; then
    results+=("$(printf '{"block":"correctness","exit_code":0,"report":"correctness_report.json"}')")
else
    results+=("$(printf '{"block":"correctness","exit_code":1,"report":"correctness_report.json"}')")
    overall_rc=1
fi


echo "========================================================"
echo "  BLOCK 2/3 — Performance (Benchmarking)"
echo "========================================================"
if run_test_block "Performance" "$DIR/test_performance.py"; then
    results+=("$(printf '{"block":"performance","exit_code":0,"report":"performance_report.json"}')")
else
    results+=("$(printf '{"block":"performance","exit_code":1,"report":"performance_report.json"}')")
    overall_rc=1
fi


echo "========================================================"
echo "  BLOCK 3/3 — Soak (Estres de Memoria)"
echo "========================================================"
if run_test_block "Soak" "$DIR/test_soak.py"; then
    results+=("$(printf '{"block":"soak","exit_code":0,"report":"soak_report.json"}')")
else
    results+=("$(printf '{"block":"soak","exit_code":1,"report":"soak_report.json"}')")
    overall_rc=1
fi


echo "╔══════════════════════════════════════════════════════════╗"
echo "║              CI SUITE — FINAL SUMMARY                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Block           Exit Code    Report"
echo "  ─────────────────────────────────────"

json_blocks="["
first=true
for entry in "${results[@]}"; do
    block=$(echo "$entry" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['block'])")
    rc=$(echo "$entry" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['exit_code'])")
    report=$(echo "$entry" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['report'])")
    status="PASS" && [ "$rc" -ne 0 ] && status="FAIL"
    printf "  %-16s %-12s %s\n" "$block" "$status" "$report"

    if [ "$first" = true ]; then first=false; else json_blocks+=","; fi
    json_blocks+="$entry"
done
json_blocks+="]"

echo "  ─────────────────────────────────────"
if [ "$overall_rc" -eq 0 ]; then
    echo "  Overall: ALL PASS"
else
    echo "  Overall: SOME BLOCKS FAILED"
fi
echo ""

timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python3 -c "
import json
summary = {
    \"status\": \"PASS\" if $overall_rc == 0 else \"FAIL\",
    \"timestamp\": \"$timestamp\",
    \"overall_exit_code\": $overall_rc,
    \"blocks\": $json_blocks,
}
with open(\"$CI_REPORT\", \"w\") as f:
    json.dump(summary, f, indent=2)
print(f'[CI] Report consolidado: $CI_REPORT')
"

exit "$overall_rc"
