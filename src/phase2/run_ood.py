"""Step 9 CLI: build/report/run/analyze the OOD (forced-choice vs explicit-fallback) experiment.

    uv run python src/phase2/run_ood.py build      # freeze data/stress/ood/
    uv run python src/phase2/run_ood.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_ood.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_ood.py analyze    # offline, from recorded results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.ood import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
