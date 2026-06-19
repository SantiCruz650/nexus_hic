#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

export NEXUS_VENV="${NEXUS_VENV:-$DIR/venv}"
VENV_PYTHON="$NEXUS_VENV/bin/python"

echo "=========================================="
echo "    COMPILANDO MOTOR NATIVO NEXUS HIC     "
echo "=========================================="

echo "[INFO] Generando datos de prueba..."
$VENV_PYTHON data/generate_data.py

echo "[INFO] Compilando backend C++..."
make clean 2>/dev/null || true
make

echo "[INFO] Ejecutando pipeline unificado..."
./nexus_hic_bridge

echo "=========================================="
echo "   NEXUS HIC: PIPELINE COMPLETADO        "
echo "=========================================="
