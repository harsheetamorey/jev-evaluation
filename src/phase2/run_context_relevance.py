"""Step 8 CLI: build/report/run/analyze the relevant-vs-irrelevant context experiment.

    uv run python src/phase2/run_context_relevance.py build      # freeze data/stress/context_relevance/
    uv run python src/phase2/run_context_relevance.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_context_relevance.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_context_relevance.py analyze    # offline; writes data/results/phase2/context/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.context_relevance import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
