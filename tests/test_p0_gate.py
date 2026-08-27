"""Gate P0 - Kernphysik gegen synthetisches Video mit bekannter Wahrheit.

Die Videos werden einmal je Modul erzeugt; der ganze Lauf muss unter 30 s bleiben.
"""

from __future__ import annotations

import subprocess

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


# --- Aufnahmerate von Hand: das verlangsamt exportierte Slo-Mo ----------
STRETCH = 8.0            # 240 fps aufgenommen, mit 30 fps abgespielt


@pytest.fixture(scope="module")
def stretched(tmp_path_factory):
    """Ein echtes, bereits verlangsamt exportiertes Video.

    Nicht simuliert: `setpts` streckt die Container-Zeitstempel genau so, wie es
    die Fotos-App beim Export einer Zeitlupe tut. Die Bilder sind dieselben, nur
    die Uhr darunter ist eine andere.
    """
    d = tmp_path_factory.mktemp("p0fps")
    orig = d / "orig240.mp4"
    truth = synth.generate(orig, h=0.40, fps=240, px_per_m=400, seed=1)
    slow = d / "slow30.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(orig),
         "-filter:v", f"setpts={STRETCH}*PTS", "-an",
         "-c:v", "libx264", "-crf", "0", "-preset", "veryfast", str(slow)],
        check=True, capture_output=True)
    return slow, truth


def test_stretched_export_is_not_measurable_as_is(stretched):
    """Ohne Angabe ist die Datei ehrlich unbrauchbar - und sagt das auch."""
    from jumpheightcalculator.physics import FitError
    path, _ = stretched
    assert abs(probe(path).measured_fps - 240 / STRETCH) < 1.0
    with pytest.raises((FitError, IngestError)):
        ball.analyze(path)


def test_capture_fps_restores_the_truth(stretched):
    """Mit der richtigen Aufnahmerate kommt dieselbe Hoehe heraus wie im Original.

    Das ist der ganze Zweck des Schalters. Er ersetzt R1 nicht: gerechnet wird
    weiter mit den echten, ungleich verteilten PTS - nur auf der richtigen Skala.
    """
    path, truth = stretched
    r = ball.analyze(path, capture_fps=240.0)
    assert abs(r.h_e2 / truth.h - 1) < 0.02, f"{r.h_e2*100:.2f} cm gegen {truth.h*100:.2f}"
    # Und die Kruemmung ist wieder g - der eigentliche Beweis, dass die Zeitachse
    # stimmt. Vorher waere sie um STRETCH^2 = 64 daneben.
    assert abs(r.a_px / truth.px_per_m - G) < 0.05


def test_wrong_capture_fps_is_wrong_by_the_square(stretched):
    """Gegenprobe: eine falsche Angabe ist quadratisch falsch, nicht harmlos.

    Der Schalter ist eine Tatsachenbehauptung des Nutzers, keine Schaetzung der
    App - und eine falsche Behauptung kann die Fit-Guete grundsaetzlich nicht
    entdecken (siehe test_nominal_fps_assumption_fails_visibly).

    **Der Schutz wirkt nur in eine Richtung.** Zu niedrig angegeben (120 statt
    240) faengt die Flugzeitgrenze ab: die Flugzeit waere 1142 ms und damit
    ausserhalb von 150-1100 ms. Zu hoch angegeben (480 statt 240) faellt durch
    jede Pruefung durch und liefert stillschweigend ein Viertel der Wahrheit -
    gemessen 10.00 gegen 40.00 cm, bei tadellosem Fit. Fuer den Sprung-Pfad
    schliesst `pipeline._check_scale_plausible` diese Luecke ueber die
    Koerpergroesse; die Kugel hat keine bekannte Groesse und damit auch keine
    Gegenprobe.
    """
    path, truth = stretched
    r = ball.analyze(path, capture_fps=480.0)      # doppelt so gross wie richtig
    assert abs(r.h_e2 / truth.h - 0.25) < 0.02, f"{r.h_e2/truth.h:.3f} statt 0.25"
    assert r.r2 > 0.99, "und der Fit sieht dabei tadellos aus"


def test_capture_fps_is_checked_on_the_corrected_axis(stretched):
    """Unsinn faellt durch dieselbe Schranke wie ein kaputtes Video (B2)."""
    path, _ = stretched
    for unsinn in (0.0, -240.0, 5.0, 100000.0):
        with pytest.raises(IngestError):
            probe(path, capture_fps=unsinn)


def test_capture_fps_keeps_the_jitter(jittered):
    """Ein konstanter Faktor darf die Ungleichmaessigkeit nicht glaetten.

    Sonst waere aus der Stauchung heimlich ein Resampling geworden - und genau
    das gleichmaessige Raster ist der Fehler, gegen den R1 gebaut ist.
    """
    path, _ = jittered
    a, b = probe(path), probe(path, capture_fps=2 * TRUE_FPS_DRIFT)
    assert b.n_frames == a.n_frames
    # Ein konstanter Faktor: jedes einzelne Frame-Intervall wird mit derselben
    # Zahl multipliziert. Welche Zahl es ist, sagt der Median der Datei (hier
    # 201.33 gemessene statt 200 nominelle fps) - geprueft wird die Konstanz.
    q = np.diff(b.t) / np.diff(a.t)
    assert np.allclose(q, q[0], rtol=1e-9), "die Stauchung ist kein Resampling"
    # Und der Vertrag: wer 400 fps sagt, bekommt eine Achse, die 400 fps misst.
    # Nicht das Doppelte der *nominellen* 200 - die Datei laeuft real mit 198.93.
    assert abs(b.measured_fps - 2 * TRUE_FPS_DRIFT) < 1e-6


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
