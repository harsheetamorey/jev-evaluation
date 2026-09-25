"""Step 4 CLI: build/report/run/analyze the ambiguity stress test.

    uv run python src/phase2/run_ambiguity.py build      # freeze data/stress/ambiguity/
    uv run python src/phase2/run_ambiguity.py report     # dry run: counts + expected API calls
    uv run python src/phase2/run_ambiguity.py run --providers jev --approve-calls N   # LIVE
    uv run python src/phase2/run_ambiguity.py analyze    # offline, from recorded results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.ambiguity import SPEC  # noqa: E402
from phase2.stress import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli(SPEC))
