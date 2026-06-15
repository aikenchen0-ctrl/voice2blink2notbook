#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from hp_acoustic_wave.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
