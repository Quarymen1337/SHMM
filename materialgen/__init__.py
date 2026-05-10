"""MaterialGen: concrete mix design with staged inverse models.

Each stage exposes a single ``run_*`` function; see ``materialgen/cli.py``
for the CLI wiring.
"""

from .train_gan import run_train_gan, finetune_generator
from .train_neat import run_train_neat
from .train_workability import run_train_workability

try:
    from .make_neat_to_bnn import run_make_neat_to_bnn
except ModuleNotFoundError:
    run_make_neat_to_bnn = None

__all__ = [
    "run_train_neat",
    "run_make_neat_to_bnn",
    "run_train_gan",
]
