"""DEPRECATED legacy evaluation entry point — hardened against test-seal bypass.

The legacy year-2024 default and the year-based loader created a test-seal
bypass (the year loader could load the sealed 2024 test events while the seal
was active) and a divergent v1 evaluation path. Both are removed.

This module now FAILS FAST and directs callers to the canonical v2 entry
points:

    python scripts/evaluate_checkpoint.py --config ... --checkpoint ... --split val --out ...
    python scripts/run_experiment.py --config ... --mode validate-only --checkpoint ...

No test IDs can be loaded or evaluated through this module.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "src/evaluation/evaluate_models.py is DEPRECATED and disabled.\n"
        "  - Validate a trained checkpoint:  python scripts/evaluate_checkpoint.py "
        "--config <cfg> --checkpoint <path> --split val --out <dir>\n"
        "  - Validate inside the runner:     python scripts/run_experiment.py "
        "--config <cfg> --mode validate-only --checkpoint <path> --out <dir>\n"
        "Test-set evaluation remains SEALED and is refused by every ordinary "
        "entry point."
    )


if __name__ == "__main__":
    sys.exit(main())
