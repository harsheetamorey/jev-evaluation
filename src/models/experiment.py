"""Experiment configuration, including a deterministic run identifier."""

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    """Identifies one evaluation run: what's being tested, on what, with which seed.

    `run_id` is derived from the other fields rather than random, so re-running the
    same experiment (same name/dataset/seed/providers) reproduces the same run_id.
    """

    name: str
    dataset: str
    providers: tuple[str, ...]
    seed: int = 0
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", self._deterministic_run_id())

    def _deterministic_run_id(self) -> str:
        basis = "|".join([self.name, self.dataset, str(self.seed), ",".join(sorted(self.providers))])
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
