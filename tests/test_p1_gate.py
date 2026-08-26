"""Gate P1 - Erkennungs-Pipeline.

P1.1 (Wiederholbarkeit) und P1.3 (Verweigerung) laufen ueber das synthetische
Video. P1.2 braucht ein echtes Sprungvideo und wird separat gefahren; die reine
Schaetzer-Algebra ist hier unabhaengig davon geprueft.
"""

from __future__ import annotations

import numpy as np
import pytest

from jumpheightcalculator import G, ball, synth
from jumpheightcalculator.ingest import IngestError
from jumpheightcalculator.physics import FitError, fit_parabola


# --- P1.1: zehn Rausch-Seeds, Streuung der Hoehe -------------------------
@pytest.mark.slow
def test_repeatability_under_noise(tmp_path):
    hs = []
    for seed in range(10):
        p = tmp_path / f"n{seed}.mp4"
        synth.generate(p, h=0.40, fps=240, px_per_m=400, noise=3.0, jitter=0.05,
                       blur=0.5, seed=seed, crf=14)
        hs.append(ball.analyze(p).h_e2)
    hs = np.array(hs)
    assert hs.std(ddof=1) < 0.01, f"sigma = {hs.std(ddof=1)*100:.3f} cm"
    assert abs(hs.mean() - 0.40) < 0.01


# --- P1.3: schlechtes Material verweigern, nicht raten -------------------
@pytest.mark.slow
def test_bad_video_is_refused(tmp_path):
    p = tmp_path / "bad.mp4"
    synth.generate(p, h=0.40, fps=30, px_per_m=400, noise=12.0, blur=1.0,
                   seed=1, crf=30)
    with pytest.raises((FitError, IngestError)):
        ball.analyze(p)


@pytest.mark.slow
def test_good_video_is_not_refused(tmp_path):
    """Gegenprobe - eine Verweigerung, die immer verweigert, ist wertlos."""
    p = tmp_path / "good.mp4"
    synth.generate(p, h=0.40, fps=240, px_per_m=400, noise=3.0, blur=0.5,
                   seed=1, crf=14)
    assert abs(ball.analyze(p).h_e2 - 0.40) < 0.01


# --- Schaetzer-Algebra, ohne Video --------------------------------------
def _synthetic_jump(h=0.45, drop=0.06, fps=240.0, px_per_m=400.0, noise=0.0, seed=0):
    """Huefte auf exakter Parabel; Landung `drop` Meter tiefer als der Absprung."""
    rng = np.random.default_rng(seed)
    v0 = np.sqrt(2 * G * h)
    # Landung tiefer: -drop = v0*t - g/2*t^2 nach t aufloesen
    t_land = (v0 + np.sqrt(v0 * v0 + 2 * G * drop)) / G
    t = np.arange(0.0, t_land, 1.0 / fps)
    y_m = v0 * t - 0.5 * G * t * t
    y_px = 1000.0 - y_m * px_per_m + rng.normal(0, noise, t.size)
    return t, y_px, v0, t_land


def test_e2_equals_v0_squared_over_2g_exactly():
    """B1 - der Grund, weshalb E3 auf anderen Landmarks fitten muss.

    Scheitelhoehe ueber dem Startpunkt und v(t0)^2/(2a) sind fuer jede Parabel
    dieselbe Zahl. Ein zweiter Schaetzer aus demselben Fit waere ein Duplikat und
    koennte per Konstruktion keinen Fehler aufdecken.
    """
    t, y, _, _ = _synthetic_jump(noise=2.0, seed=3)
    f = fit_parabola(t, y, seed=0)
    s = f.scale
    apex_over_start = (float(f.y_at(t[0])) - f.y_apex) * s
    from_v0 = (float(f.v_up_px(t[0])) * s) ** 2 / (2 * G)
    assert abs(apex_over_start - from_v0) < 1e-12


def test_flight_time_estimator_overestimates_when_landing_is_lower():
    """R2 - E1 muss ueberschaetzen, weil man gestreckt abspringt und gebeugt landet."""
    h, drop = 0.45, 0.06
    t, y, v0, t_land = _synthetic_jump(h=h, drop=drop)
    f = fit_parabola(t, y, seed=0)
    s = f.scale
    h_e1 = G * t_land ** 2 / 8.0
    h_e2 = (float(f.y_at(0.0)) - f.y_apex) * s
    assert h_e2 == pytest.approx(h, abs=1e-3)
    assert h_e1 > h_e2, "E1 muss ueber E2 liegen"
    # Der Ueberschuss ist rund die halbe Absenkung - hier gut 3 cm, genau die
    # Groessenordnung, die die Literatur fuer die Flugzeitmethode nennt.
    assert 0.015 < h_e1 - h_e2 < 0.045


def test_delta_recovers_the_landing_drop():
    """Delta = (v_land^2 - v0^2)/(2g) misst die Absenkung des Schwerpunkts."""
    drop = 0.06
    t, y, _, t_land = _synthetic_jump(drop=drop)
    f = fit_parabola(t, y, seed=0)
    s = f.scale
    v0 = float(f.v_up_px(0.0)) * s
    v_land = abs(float(f.v_up_px(t_land))) * s
    assert (v_land ** 2 - v0 ** 2) / (2 * G) == pytest.approx(drop, abs=1e-3)


def test_e4_equals_e2_plus_delta():
    """E4 nutzt nur die Fallphase, liegt also systematisch um Delta hoeher."""
    h, drop = 0.45, 0.06
    t, y, _, t_land = _synthetic_jump(h=h, drop=drop)
    f = fit_parabola(t, y, seed=0)
    s = f.scale
    h_e2 = (float(f.y_at(0.0)) - f.y_apex) * s
    h_e4 = 0.5 * G * max(t_land - f.t_apex, 0.0) ** 2
    assert h_e4 == pytest.approx(h_e2 + drop, abs=1e-3)


def test_cubic_test_is_deflated_by_autocorrelation():
    """B3 - der Verzeichnungstest darf an korrelierten Residuen nicht anschlagen.

    Ohne die Korrektur meldete er auf einer sauberen echten Flugbahn t = 4.5 und
    verweigerte die Messung.
    """
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 0.6, 1 / 240)
    clean = 1000.0 - (3.0 * t - 0.5 * G * t * t) * 400.0
    white = clean + rng.normal(0, 2.0, t.size)
    ar = clean.copy()
    e = 0.0
    for i in range(t.size):                     # stark autokorreliertes Rauschen
        e = 0.9 * e + rng.normal(0, 2.0)
        ar[i] += e
    assert fit_parabola(t, white, seed=0).cubic_t < 4.0
    assert fit_parabola(t, ar, seed=0).cubic_t < 4.0


# --- Perspektive: Massstabsaenderung waehrend des Flugs ---------------------
def _track_with_drift(drift: float):
    """Kuenstlicher Track; `drift` ist die Groessenaenderung starrer Segmente."""
    from jumpheightcalculator import pipeline as pl
    n, n0, n1, fps = 220, 40, 140, 120.0
    t = np.arange(n) / fps
    v0 = float(np.sqrt(2 * G * 0.45))
    y = np.full(n, 1000.0)
    u = t[n0:n1] - t[n0]
    y[n0:n1] = 1000.0 - (v0 * u - 0.5 * G * u * u) * 400.0
    contact = np.ones(n); contact[n0:n1] = 0.0
    return pl.TrackData(
        t=t, xy=np.zeros((n, 33, 2), np.float32), width=540, height=960,
        hip_y=y, com_y=y, ankle_y=y + 150, contact=contact,
        confidence=np.full(n, 0.9), n0=n0, n1=n1, stance_end=n0 - 30, fps=fps,
        shift_residual=0.1, body_scale_px=400.0, scale_drift=drift)


def test_perspective_drift_is_refused():
    """Wer auf die Kamera zu springt, aendert den Massstab mitten im Sprung.

    Gemessen an einem echten Video: die Oberschenkel wurden im Flug 33 % laenger
    im Bild. Ein Knochen waechst nicht - das ist Tiefenbewegung, und damit ist
    weder die Kruemmung noch die Selbstkalibrierung (R5) noch zu gebrauchen.
    """
    from jumpheightcalculator.pipeline import Refusal, measure
    with pytest.raises(Refusal, match="grösser im Bild"):
        measure(_track_with_drift(0.33))


def test_small_drift_is_accepted():
    """Gegenprobe - ein sauberes Video streut rund 3 %, das darf nicht stoeren."""
    from jumpheightcalculator.pipeline import measure
    r = measure(_track_with_drift(0.03))
    assert r.h_e2 == pytest.approx(0.45, abs=0.02)
