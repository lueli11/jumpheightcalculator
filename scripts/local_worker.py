"""Worker ohne Redis - für Entwicklungsmaschinen, auf denen kein Redis läuft.

Pollt die Datenbank nach `queued`-Analysen und ruft dieselbe `run_analysis`, die
auch arq aufruft. In Produktion läuft stattdessen `arq jumpheightcalculator.worker.WorkerSettings`;
dieser Weg existiert nur, damit sich der Review-Screen lokal end-to-end testen
lässt, wo Docker und WSL fehlen.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jumpheightcalculator import db, worker  # noqa: E402

DATA = Path(os.environ.get("JUMPHEIGHTCALCULATOR_DATA", "data"))


def main() -> None:
    print(f"lokaler Worker auf {DATA}", flush=True)
    while True:
        con = db.connect(DATA / "jumpheightcalculator.sqlite")
        row = con.execute(
            "SELECT id FROM analysis WHERE status='queued' ORDER BY created_at"
        ).fetchone()
        con.close()
        if row is None:
            time.sleep(1.0)
            continue
        print(f"-> {row['id']}", flush=True)
        t0 = time.time()
        state = worker.run_analysis(row["id"], data_dir=DATA)
        # "skipped" heisst: ein anderer Worker war schneller. Kein Fehler.
        print(f"   {state} nach {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
