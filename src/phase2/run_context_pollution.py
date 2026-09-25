"""Step 7 CLI: build/report/run/analyze the context-pollution experiment.

    uv run python src/phase2/run_context_pollution.py build      # freeze data/stress/context_pollution/
    uv run python src/phase2/run_context_pollution.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_context_pollution.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_context_pollution.py analyze    # offline, from recorded results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.context_pollution import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
