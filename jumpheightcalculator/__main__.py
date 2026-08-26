"""CLI. Erst wenn die Zahlen an einem Video stimmen, kommt HTTP drumherum."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import G, ball, pipeline, synth
from .ingest import IngestError, probe
from .physics import FitError
from .pipeline import Refusal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jumpheightcalculator", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="synthetisches Testvideo mit bekannter Wahrheit")
    s.add_argument("out")
    s.add_argument("--height-m", type=float, default=0.40)
    s.add_argument("--fps", type=float, default=240.0)
    s.add_argument("--px-per-m", type=float, default=400.0)
    s.add_argument("--jitter", type=float, default=0.0, help="Frame-Abstand +- Anteil")
    s.add_argument("--noise", type=float, default=0.0, help="Pixelrauschen [Grauwerte]")
    s.add_argument("--blur", type=float, default=0.0, help="Belichtung als Anteil von dt")
    s.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("validate", help="Pipeline gegen reine Physik prüfen")
    vs = v.add_subparsers(dest="kind", required=True)
    vb = vs.add_parser("ball", help="Ballwurf-Test: die Krümmung muss g ergeben")
    vb.add_argument("video")
    vb.add_argument("--px-per-m", type=float, default=None,
                    help="bekannter Massstab; nur zum Gegenrechnen der Krümmung")
    vb.add_argument("--json", action="store_true")

    an = sub.add_parser("analyze", help="Sprung aus einem Video messen")
    an.add_argument("video")
    an.add_argument("--jump-type", default="cmj", choices=["cmj", "sj", "drop", "hops"])
    an.add_argument("--mass-kg", type=float, default=None,
                    help="für Impuls und mittlere Kraft; ohne Masse entfallen beide")
    an.add_argument("--model", default="full", choices=["full", "heavy"])
    an.add_argument("--scale", type=float, default=None,
                    help="Arbeitsauflösung als Anteil der Originalgrösse; "
                         "ohne Angabe automatisch auf 540 px Breite")
    an.add_argument("--json", action="store_true")

    i = sub.add_parser("info", help="Zeitstempel und Geometrie eines Videos")
    i.add_argument("video")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "synth":
            t = synth.generate(a.out, h=a.height_m, fps=a.fps, px_per_m=a.px_per_m,
                               jitter=a.jitter, noise=a.noise, blur=a.blur, seed=a.seed)
            print(json.dumps(asdict(t), indent=2))
            return 0

        if a.cmd == "analyze":
            r = pipeline.analyze(a.video, jump_type=a.jump_type, mass_kg=a.mass_kg,
                                 model=a.model, scale=a.scale)
            if a.json:
                print(json.dumps(r.to_dict(), indent=2))
                return 0
            print(f"Sprunghöhe (Referenz E2) {r.h_e2*100:8.1f} cm")
            print(f"  E1 aus der Flugzeit     {r.h_e1*100:8.1f} cm   "
                  f"(überschätzt um Bauart, R2)")
            print(f"  E3 Schwerpunkt          {r.h_e3*100:8.1f} cm   "
                  f"Uneinigkeit {r.disagreement*100:.1f} cm")
            print(f"  E4 nur Fallphase        {r.h_e4*100:8.1f} cm")
            print()
            print(f"Absprunggeschwindigkeit {r.v0:8.2f} m/s")
            print(f"Flugzeit                {r.t_f*1000:8.0f} ms")
            print(f"Landeabsenkung Delta    {r.delta*100:+8.1f} cm   "
                  f"(so viel liegt E1 bei dir zu hoch)")
            print(f"Abdruckdauer            {r.push_time*1000:8.0f} ms")
            if r.contact_time:
                print(f"Bodenkontaktzeit        {r.contact_time*1000:8.0f} ms")
            if r.rsi:
                print(f"RSI                     {r.rsi:8.2f}")
            if r.impulse:
                print(f"Impuls                  {r.impulse:8.1f} Ns")
            if r.mean_force:
                print(f"mittlere Kraft          {r.mean_force:8.0f} N")
            print()
            print(f"Absprungframe {r.takeoff_frame:.2f}   Landeframe {r.landing_frame:.2f}"
                  f"   Flugframes {r.n_flight_frames}")
            print(f"Güte {r.quality:.2f}   R2 {r.r2_hip:.4f}   {r.fps_measured:.0f} fps   "
                  f"Pipeline {r.pipeline_version}")
            for w in r.warnings:
                print(f"WARNUNG: {w}")
            return 0

        if a.cmd == "info":
            info = probe(a.video)
            print(f"{info.path.name}: {info.width}x{info.height}, Rotation {info.rotation} Grad")
            print(f"  {info.n_frames} Frames, {info.duration:.3f} s")
            print(f"  gemessene fps {info.measured_fps:.2f}  (Container meldet "
                  f"{info.nominal_fps:.2f})")
            if info.measured_fps < 120:
                print("  WARNUNG: unter 120 fps. Für eine Fortschrittsmessung zu grob - "
                      "bitte in Slo-Mo aufnehmen.")
            return 0

        if a.cmd == "validate" and a.kind == "ball":
            r = ball.analyze(a.video)
            d = r.to_dict()
            if a.px_per_m:
                d["curvature_ms2"] = r.a_px / a.px_per_m
            if a.json:
                print(json.dumps(d, indent=2))
            else:
                print(f"Höhe (Scheitel)     {r.h_e2*100:8.2f} cm")
                print(f"Höhe (Flugzeit)     {r.h_e1*100:8.2f} cm")
                print(f"Flugzeit             {r.t_f*1000:8.1f} ms")
                print(f"v0                   {r.v0:8.3f} m/s")
                print(f"Massstab (aus g)     {1/r.scale:8.1f} px/m")
                if a.px_per_m:
                    c = r.a_px / a.px_per_m
                    print(f"Krümmung            {c:8.3f} m/s^2   "
                          f"(soll {G}, Abweichung {c-G:+.3f})")
                print(f"R2 {r.r2:.5f}  RMS {r.rms_resid_px:.2f} px  "
                      f"kubisch t={r.cubic_t:.2f}  Flugframes {r.n_flight_frames}")
            return 0
    except (IngestError, FitError, Refusal) as e:
        # R7: Grund nennen, keine Zahl erfinden.
        print(f"Keine Messung möglich: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
