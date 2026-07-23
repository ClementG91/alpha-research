from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

import research


if __name__ == "__main__":
    try:
        asyncio.run(research.main())
    except Exception:
        out = Path("results")
        out.mkdir(exist_ok=True)
        (out / "FATAL.txt").write_text(traceback.format_exc())
        raise
