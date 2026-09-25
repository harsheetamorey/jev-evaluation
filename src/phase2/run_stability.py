"""Step 5 CLI: build/report/run/analyze the semantic-stability experiment.

    uv run python src/phase2/run_stability.py build      # freeze data/stress/stability/
    uv run python src/phase2/run_stability.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_stability.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_stability.py analyze    # offline, from recorded results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.stability import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
