"""Step 10 CLI: build/report/run/analyze the adversarial-injection suite.

    uv run python src/phase2/run_adversarial.py build      # freeze data/stress/adversarial/
    uv run python src/phase2/run_adversarial.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_adversarial.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_adversarial.py analyze    # offline; writes data/results/phase2/adversarial/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.adversarial import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
