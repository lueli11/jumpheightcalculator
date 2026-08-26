"""Gate P0 - Kernphysik gegen synthetisches Video mit bekannter Wahrheit.

Die Videos werden einmal je Modul erzeugt; der ganze Lauf muss unter 30 s bleiben.
"""

from __future__ import annotations

import numpy as np
import pytest

from jumpheightcalculator import G, ball, synth
from jumpheightcalculator.ingest import IngestError, probe

TRUE_FPS_DRIFT = 200.0     # was die Datei wirklich tut
ASSUMED_FPS = 240.0        # was ein naives Werkzeug dem Aufnahmemodus glaubt


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """240 fps, rauschfrei, gleichmaessig getaktet."""
    p = tmp_path_factory.mktemp("p0") / "clean240.mp4"
    return p, synth.generate(p, h=0.40, fps=240, px_per_m=400, seed=1)


@pytest.fixture(scope="module")
def jittered(tmp_path_factory):
    """Echte 200 fps mit +-15 % Frame-Abstand."""
    p = tmp_path_factory.mktemp("p0") / "jitter200.mp4"
    return p, synth.generate(p, h=0.40, fps=TRUE_FPS_DRIFT, px_per_m=400,
                             jitter=0.15, seed=3)


# --- P0.1 ---------------------------------------------------------------
def test_curvature_is_gravity(clean):
    """Die gefittete Kruemmung muss g ergeben.

    Geprueft wird gegen den *bekannten* Massstab des synthetischen Videos, nicht
    ueber die Selbstkalibrierung: die liefert per Konstruktion 9.81 und wuerde
    damit gar nichts messen.
    """
    path, truth = clean
    r = ball.analyze(path)
    curvature = r.a_px / truth.px_per_m          # px/s^2 / (px/m) = m/s^2
    assert abs(curvature - G) < 0.05, f"{curvature:.4f} m/s^2"


# --- P0.2 ---------------------------------------------------------------
def test_height_within_one_percent(clean):
    path, truth = clean
    r = ball.analyze(path)
    assert abs(r.h_e2 / truth.h - 1) < 0.01, f"{r.h_e2*100:.3f} cm"


def test_flight_time_and_v0(clean):
    path, truth = clean
    r = ball.analyze(path)
    assert abs(r.t_f / truth.t_f - 1) < 0.01
    assert abs(r.v0 / truth.v0 - 1) < 0.01


def test_scale_recovered_from_physics_alone(clean):
    """R5 - der Massstab kommt ohne Massband und ohne Referenzobjekt heraus."""
    path, truth = clean
    r = ball.analyze(path)
    assert abs((1 / r.scale) / truth.px_per_m - 1) < 0.01


# --- P0.3a: Robustheit gegen variable Framerate --------------------------
def test_variable_framerate_within_two_percent(jittered):
    path, truth = jittered
    r = ball.analyze(path)
    assert abs(r.h_e2 / truth.h - 1) < 0.02, f"{r.h_e2*100:.3f} cm"


# --- P0.3b: Negativprobe - ohne echte Zeitstempel muss es danebenliegen ---
def test_nominal_fps_assumption_fails_visibly(jittered):
    """Beweist, dass R1 wirklich traegt.

    Ohne diesen Gegenlauf zeigt P0.3a nur, dass irgendetwas funktioniert - nicht,
    dass die Containerzeitstempel dafuer noetig waren. Der gleichverteilte Jitter
    allein genuegt als Beweis nicht, weil er mittelwertfrei ist: eine angenommene
    Framerate waere im Mittel trotzdem richtig. Erst wenn die angenommene
    Framerate von der tatsaechlichen abweicht - der Alltagsfall, das Handy
    schreibt "240 fps Slo-Mo" und liefert 200 - wird der Fehler systematisch.
    """
    path, truth = jittered
    good = ball.analyze(path)
    naive = ball.analyze(path, assume_fps=ASSUMED_FPS)

    assert abs(good.h_e2 / truth.h - 1) < 0.02
    assert abs(naive.h_e2 / truth.h - 1) > 0.10, (
        f"Die Negativprobe liegt bei {naive.h_e2*100:.2f} cm zu nahe an der "
        f"Wahrheit - der Test belegt dann nicht, dass die Zeitstempel tragen."
    )
    # Der Fehler steckt in der Zeitachse, nicht in der Bahn: die falsche Rechnung
    # rekonstruiert eine Erdbeschleunigung, die keine ist ...
    assert abs(naive.a_px / truth.px_per_m - G) > 1.0
    # ... und ihr R^2 ist trotzdem ausgezeichnet. Ein Zeitbasis-Fehler ist an der
    # Fit-Guete grundsaetzlich nicht erkennbar, nur an der Physik.
    assert naive.r2 > 0.99


# --- B2: unplausible Zeitbasis wird verweigert, nicht geraten ------------
def test_implausible_timebase_is_refused(tmp_path):
    p = tmp_path / "slow.mp4"
    synth.generate(p, h=0.40, fps=5, px_per_m=400, seed=0)
    with pytest.raises(IngestError, match="Framerate"):
        probe(p)


# --- Determinismus: Voraussetzung fuer Gate P2.1 -------------------------
def test_analysis_is_reproducible(clean):
    path, _ = clean
    a, b = ball.analyze(path), ball.analyze(path)
    assert a.to_dict() == b.to_dict()
