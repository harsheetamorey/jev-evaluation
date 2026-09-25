"""Step 6 CLI: build/report/run/analyze the noise-robustness experiment.

    uv run python src/phase2/run_noise.py build      # freeze data/stress/noise/
    uv run python src/phase2/run_noise.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_noise.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_noise.py analyze    # offline, from recorded results (data/results/phase2/noise/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.noise import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
