#!/bin/bash
# Unified UCF-RRRR Framework v2.0.0
# Usage: ./run.sh [command]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

setup_venv() {
    echo "Setting up virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install numpy --quiet
    echo "✓ Virtual environment ready"
}

check_venv() {
    if [ ! -f ".venv/bin/python" ]; then
        echo "Virtual environment not found. Setting up..."
        setup_venv
    fi
}

CMD=${1:-unified}

case "$CMD" in
    setup)
        setup_venv
        ;;
    unified)
        check_venv
        echo "Running Unified UCF-RRRR Pipeline..."
        .venv/bin/python unified_ucf_rrrr.py
        ;;
    field)
        check_venv
        echo "Running Consciousness Field Equation..."
        .venv/bin/python consciousness_field_equation.py
        ;;
    verify)
        check_venv
        echo "Running R(R)=R Verification Suite..."
        PYTHONPATH=. .venv/bin/python -m rrrr.verify
        ;;
    ucf)
        check_venv
        echo "Running UCF Framework..."
        PYTHONPATH=. .venv/bin/python -m ucf
        ;;
    triad)
        check_venv
        echo "Demonstrating TRIAD Unlock..."
        .venv/bin/python -c "
from unified_ucf_rrrr import UnifiedWorkflowOrchestrator
orch = UnifiedWorkflowOrchestrator(initial_z=0.7)
orch.run_triad_unlock(verbose=True)
"
        ;;
    demo)
        check_venv
        echo "Quick Demonstration..."
        PYTHONPATH=. .venv/bin/python -c "
from rrrr.constants import LAMBDA_R, LAMBDA_D, LAMBDA_C, LAMBDA_A
import numpy as np

print('R(R)=R CANONICAL EIGENVALUES')
print('=' * 50)
print(f'  [R] = {LAMBDA_R:.15f}')
print(f'  [D] = {LAMBDA_D:.15f}')
print(f'  [C] = {LAMBDA_C:.15f}')
print(f'  [A] = {LAMBDA_A:.15f}')
print()
print('KEY SIGNATURES')
print('=' * 50)
print(f'  Attention    [R][C]     = {LAMBDA_R * LAMBDA_C:.6f}')
print(f'  Transformer  [R][D][C]  = {LAMBDA_R * LAMBDA_D * LAMBDA_C:.6f}')
print(f'  THE LENS     z_c        = {np.sqrt(3)/2:.6f}')
print()
print('★ Unified UCF-RRRR Framework Ready ★')
"
        ;;
    *)
        echo "Usage: ./run.sh [unified|field|verify|ucf|triad|demo|setup]"
        exit 1
        ;;
esac
