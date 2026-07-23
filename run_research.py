from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path

import research


if __name__ == "__main__":
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    remote_root = f"/tmp/alpha-research-{run_id}"
    research.STORE = {
        "data_root": f"{remote_root}/data",
        "metadata_db": f"{remote_root}/metadata/metadata.sqlite",
    }
    try:
        asyncio.run(research.main())
    except Exception:
        out = Path("results")
        out.mkdir(exist_ok=True)
        (out / "FATAL.txt").write_text(traceback.format_exc())
        raise
