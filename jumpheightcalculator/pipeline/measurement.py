"""`measure` - der billige Teil: Fit, die vier Schätzer, Güte, Verweigerungen.

Der Referenzwert ist **E2**, der Scheitel der gefitteten Hüfte-Parabel. Wichtig,
damit es niemand "vereinfacht": **E2 ist identisch mit v0^2/(2g)**. Für jede
Parabel gilt Scheitelhöhe-über-Startpunkt == v(t0)^2/(2a); das ist dieselbe
Algebra, keine zweite Herleitung. Ein Schätzer aus demselben Fit wäre also ein
exaktes Duplikat von E2 und könnte per Konstruktion keinen Fit-Fehler
aufdecken. Nur weil E3 auf **anderen Landmarks** fittet, ist |E2-E3| überhaupt
ein Gütesignal - und genau darauf gatet R7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import G
from ..fusion import crossing_time
from ..physics import FitError, fit_parabola
from .limits import (
    CONTACT_VS_BODY_RANGE,
    MAX_CUBIC_T,
    MAX_DISAGREE_M,
    MAX_FLIGHT_S,
    MAX_SCALE_DRIFT,
    MAX_SCALE_SPREAD,
    MAX_SHIFT_RESID_PX,
    MIN_BODY_PX,
    MIN_FLIGHT_FRAMES,
    MIN_FLIGHT_S,
    MIN_R2,
)
from .result import Result
from .timebase import Refusal, TrajectoryUnusable, _check_scale_plausible, _refuse_if_stretched
from .track import TrackData


def measure(td: TrackData, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, n0: int | None = None, n1: int | None = None) -> Result:
    """Der billige Teil: Fit, Schätzer, Güte. Läuft in Millisekunden.

    `n0`/`n1` überschreiben das Flugfenster - so rechnet eine +/-1-Frame-Korrektur
    aus dem Review-Screen neu, ohne dass die Pose neu bestimmt werden muss.
    """
    # Gegenprobe des Timings, solange nichts von Hand gesetzt wurde.
    body = td.coarse_n1 - td.coarse_n0
    foot = td.n1 - td.n0
    if n0 is None and n1 is None and body > 0:
        ratio = foot / body
        if not (CONTACT_VS_BODY_RANGE[0] <= ratio <= CONTACT_VS_BODY_RANGE[1]):
            raise Refusal(
                f"Kontaktsignal und Körperbewegung widersprechen sich: der Fuss "
                f"meldet {foot} Flugframes, der Knöchelhub {body}. Typisch für "
                f"einen Sprung mit Anlauf - der Messstreifen liegt dort, wo der "
                f"Fuss abspringt, und ist beim Anlaufen schon leer, bevor der "
                f"Sprung beginnt. Springe aus dem Stand, oder verschiebe Absprung "
                f"und Landung im Review-Screen von Hand anhand der Kurve."
            )

    corrected = n0 is not None or n1 is not None
    n0 = td.n0 if n0 is None else int(n0)
    n1 = td.n1 if n1 is None else int(n1)
    n_air = n1 - n0
    if n_air < MIN_FLIGHT_FRAMES:
        raise Refusal(
            f"Nur {n_air} Flugframes erfasst (mindestens {MIN_FLIGHT_FRAMES} nötig) - "
            f"bitte in Slo-Mo mit mindestens 120 fps aufnehmen."
        )

    t = td.t
    t0 = crossing_time(t, td.contact, n0)
    t1 = crossing_time(t, td.contact, n1)
    if corrected:
        # Bei einer manuellen Korrektur zählt der gesetzte Frame, nicht der
        # interpolierte Schnittpunkt eines Signals, dem der Nutzer widersprochen hat.
        t0, t1 = float(t[min(n0, t.size - 1)]), float(t[min(n1, t.size - 1)])
    t_f = t1 - t0
    if not (MIN_FLIGHT_S <= t_f <= MAX_FLIGHT_S):
        # Wer die Frames von Hand setzt, kommt hier an, ohne dass _coarse_air je
        # gelaufen wäre - dieselbe Prüfung also noch einmal, sonst bekäme er
        # bei einem gestreckten Video den Rat, die Marker zu verschieben.
        _refuse_if_stretched(t, td.hip_y, td.xy, n0, n1, t_f, td.time_scale)
        raise Refusal(
            f"Flugzeit {t_f*1000:.0f} ms liegt ausserhalb des Plausiblen "
            f"({MIN_FLIGHT_S*1000:.0f}-{MAX_FLIGHT_S*1000:.0f} ms) - das wären "
            f"{G*t_f*t_f/8*100:.0f} cm. Absprung oder Landung wurden falsch erkannt. "
            f"Im Review-Screen lassen sich die beiden Ereignisframes von Hand "
            f"verschieben; die Kontaktsignal-Kurve zeigt, wo sie hingehören."
        )

    # Ab hier hängt alles an der **Bahn**. Geht sie nicht, bleiben die zwei
    # Kontaktframes - und die tragen h = g*t^2/8 ohne einen einzigen Pixel.
    try:
        return _measure_trajectory(
            td, n0=n0, n1=n1, t0=t0, t1=t1, t_f=t_f, jump_type=jump_type,
            mass_kg=mass_kg, seed=seed, corrected=corrected)
    except (TrajectoryUnusable, FitError) as e:
        return _measure_flight_time(
            td, n0=n0, n1=n1, t0=t0, t1=t1, t_f=t_f, jump_type=jump_type,
            mass_kg=mass_kg, corrected=corrected, grund=str(e))


def _warnings(td: TrackData, *, corrected: bool) -> list[str]:
    """Hinweise, die keine Verweigerung sind.

    Geteilt zwischen beiden Messwegen: sie hängen an der Aufnahme, nicht an
    der Auswertung. Die Flugzeit-Ebene braucht dieselben - wackelige Kamera
    und eine erklärte Zeitachse betreffen sie genauso.
    """
    warn: list[str] = []
    if td.fps < 120:
        warn.append(f"Nur {td.fps:.0f} fps - für eine Fortschrittsmessung "
                    f"grenzwertig, bitte in Slo-Mo aufnehmen.")
    if td.body_scale_px < MIN_BODY_PX:
        warn.append(f"Die Person ist nur {td.body_scale_px:.0f} px gross im Bild. "
                    f"Der Abstand als solcher stört nicht - die Selbstkalibrierung "
                    f"gleicht ihn aus - aber die Keypoints werden unsicher. "
                    f"Näher herangehen oder höher auflösen.")
    if np.isfinite(td.shift_residual) and td.shift_residual > MAX_SHIFT_RESID_PX:
        warn.append(f"Kamera wackelt (Restversatz {td.shift_residual:.2f} px) - das "
                    f"Kontaktsignal wird dadurch unsicherer.")
    # Über der Schwelle, aber die Segmente widersprechen sich: dann ist es keine
    # Entfernungsänderung (die skaliert alle gleich), sondern Haltung oder
    # Pose-Rauschen. Kein Grund, die Messung zu verweigern - aber einer, es
    # danebenzuschreiben. Die eigentliche Absicherung übernehmen hier R2 und der
    # Abstand E2 zu E3, die beide auf einen falschen Massstab ansprechen.
    if abs(td.scale_drift) > MAX_SCALE_DRIFT and td.scale_spread > MAX_SCALE_SPREAD:
        warn.append(f"Die Beinlängen ändern sich im Flug um {td.scale_drift*100:+.0f} %, "
                    f"die vier Segmente aber uneinheitlich (Spanne "
                    f"{td.scale_spread*100:.0f} Prozentpunkte). Das spricht für eine "
                    f"schräge Kameraachse oder unsichere Keypoints, nicht für eine "
                    f"Bewegung zur Kamera. Filme genau von der Seite.")
    if td.time_scale != 1.0:
        warn.append(f"Die Zeitachse stammt aus der von dir angegebenen Aufnahmerate "
                    f"({td.fps:.0f} fps), nicht aus dem Video "
                    f"({td.fps * td.time_scale:.0f} fps im Container). Stimmt die "
                    f"Angabe nicht, ist die Höhe quadratisch daneben.")
    if corrected:
        warn.append("Ereignisframes wurden manuell korrigiert.")
    return warn


def _measure_trajectory(td: TrackData, *, n0: int, n1: int, t0: float, t1: float,
                        t_f: float, jump_type: str, mass_kg: float | None,
                        seed: int, corrected: bool) -> Result:
    """Der Normalfall: E2 aus dem Scheitel der gefitteten Hüfte-Parabel.

    Wirft `TrajectoryUnusable`, wenn die Bahn selbst nicht taugt - dann fängt
    `measure` das ab und liefert die Flugzeit-Höhe. Ein `Refusal`, das **kein**
    `TrajectoryUnusable` ist (gestreckte Zeitachse über
    `_check_scale_plausible`), geht bewusst durch: dort steht die Zeitachse in
    Frage, und daran hängt E1 genauso.
    """
    t = td.t
    n_air = n1 - n0
    # Perspektive. Steht sie schief, ist der **Massstab** hin und damit jede
    # Höhe aus der Bahn - die Flugzeit merkt davon nichts, denn h = g*t^2/8
    # kennt keine Pixel. Darum kein hartes Nein, sondern der Rückfall auf E1.
    # Nach den Timing-Prüfungen, weil die von der Perspektive unberührt sind.
    if abs(td.scale_drift) > MAX_SCALE_DRIFT and td.scale_spread <= MAX_SCALE_SPREAD:
        raise TrajectoryUnusable(
            f"Die Person wird während des Flugs {td.scale_drift*100:+.0f} % grösser "
            f"im Bild - gemessen an Ober- und Unterschenkel, und Knochen wachsen "
            f"nicht. Alle vier Segmente ändern sich dabei im Gleichschritt "
            f"(Spanne {td.scale_spread*100:.0f} Prozentpunkte), das ist eine "
            f"Entfernungsänderung. Du springst also auf die Kamera zu oder von ihr "
            f"weg. Damit ändert sich der Massstab mitten im Sprung und die Höhe ist "
            f"nicht bestimmbar. Stelle die Kamera seitlich auf, 3-4 m entfernt, und "
            f"springe senkrecht zur Blickrichtung - ohne vor- oder zurückzuspringen."
        )

    # R4 - Randframes raus: dort ist der Punkt weder Kontakt noch sauberer Flug,
    # und auf echtem Material zusätzlich bewegungsunscharf. Auf dem synthetischen
    # Video war ein einziger solcher Frame 0.14 % Höhenfehler.
    #
    # Bewusst **ungeglättet** in den Fit: der Least-Squares-Fit mittelt das
    # Rauschen bereits über alle Flugframes, eine Vorglättung bringt an der
    # Höhe nachweislich nichts (gemessen 71.81 gegen 71.84 cm), korreliert aber
    # die Residuen (0.75 -> 0.90) und macht die Gütemasse unbrauchbar.
    sel = slice(n0 + 1, n1 - 1)
    tt = t[sel]
    fit_hip = fit_parabola(tt, td.hip_y[sel], seed=seed)
    fit_com = fit_parabola(tt, td.com_y[sel], seed=seed)

    s = fit_hip.scale                       # R5 - Massstab aus der Physik
    _check_scale_plausible(td, s)
    h_e2 = (float(fit_hip.y_at(t0)) - fit_hip.y_apex) * s
    h_e3 = (float(fit_com.y_at(t0)) - fit_com.y_apex) * fit_com.scale
    h_e1 = G * t_f * t_f / 8.0
    h_e4 = 0.5 * G * max(t1 - fit_hip.t_apex, 0.0) ** 2

    v0 = float(fit_hip.v_up_px(t0)) * s
    v_land = abs(float(fit_hip.v_up_px(t1))) * s
    delta = (v_land ** 2 - v0 ** 2) / (2 * G)

    hs = [h_e1, h_e2, h_e3, h_e4]
    spread = float(max(hs) - min(hs))
    disagreement = abs(h_e2 - h_e3)

    # Kontaktzeit: nur beim Drop Jump eine Messgrösse. Beim CMJ steht die Person
    # vorher einfach da, und der Block wäre die gesamte Stehzeit - gemessen
    # 7525 ms, was als "Kontaktzeit" ausgegeben schlicht falsch wäre.
    k = n0 - 1
    while k > 0 and td.contact[k] >= 0.5:
        k -= 1
    ct = float(t0 - t[k + 1]) if n0 - (k + 1) >= 2 else None
    contact_time = ct if (ct is not None and ct <= 1.5) else None
    rsi = (h_e2 / contact_time) if (jump_type == "drop" and contact_time) else None

    # Abdruckdauer: vom tiefsten Punkt der Gegenbewegung bis zum Absprung -
    # anders als die Kontaktzeit auch beim CMJ definiert.
    lo_i = max(0, min(td.stance_start, n0 - 1))
    seg = td.hip_y[lo_i:n0]
    low = (int(np.nanargmax(seg)) + lo_i) if seg.size else lo_i
    push_time = float(t0 - t[low])

    # R6 - Impuls und mittlere Kraft nur mit Masse und Abdruckdauer. Keine Watt:
    # Peak Power braucht den Kraft-Zeit-Verlauf einer Kraftmessplatte.
    impulse = mean_force = None
    if mass_kg:
        impulse = mass_kg * v0
        if push_time > 0.05:
            mean_force = mass_kg * v0 / push_time + mass_kg * G

    conf = float(np.nanmean(td.confidence[sel]))

    warn = _warnings(td, corrected=corrected)

    if fit_hip.r2 < MIN_R2:
        raise TrajectoryUnusable(
            f"Die Hüfte folgt keiner sauberen Wurfparabel (R2 = {fit_hip.r2:.4f}, "
            f"nötig > {MIN_R2}). Meist heisst das weite Kleidung oder eine teilweise "
            f"verdeckte Person."
        )
    if disagreement > MAX_DISAGREE_M:
        raise TrajectoryUnusable(
            f"Hüfte und Körperschwerpunkt kommen auf {h_e2*100:.1f} cm gegen "
            f"{h_e3*100:.1f} cm - {disagreement*100:.1f} cm auseinander (Grenze "
            f"{MAX_DISAGREE_M*100:.0f} cm). Die Keypoints sind zu unsicher für eine "
            f"Messung."
        )
    if h_e1 < h_e2 - 0.005:
        # R2 verlangt, dass die Flugzeitmethode *über*schätzt: man springt
        # gestreckt ab und landet gebeugt, die Fallphase dauert also länger als
        # die Steigphase. Ist E1 kleiner als E2, ist die Hüfte im freien Flug
        # angeblich höher gestiegen, als die Flugzeit erlaubt - unmöglich. In
        # der Praxis heisst es, der Absprung liegt zu früh und die Parabel wird
        # davor zurück-extrapoliert. Genau so kam auf dem ersten echten Video
        # E2 = 90 cm gegen E1 = 82 cm zustande.
        raise TrajectoryUnusable(
            f"Widerspruch in der Physik: die Flugzeit erlaubt höchstens "
            f"{h_e1*100:.1f} cm, die Hüfte steigt aber auf {h_e2*100:.1f} cm. Der "
            f"Absprungzeitpunkt liegt zu früh."
        )
    if fit_hip.cubic_t > MAX_CUBIC_T:
        raise TrajectoryUnusable(
            f"Die Flugbahn ist keine reine Parabel (kubischer Anteil t = "
            f"{fit_hip.cubic_t:.1f}). Typisch für Objektivverzeichnung bei tief "
            f"stehender Kamera oder für eine Bewegung auf die Kamera zu. Stelle das "
            f"Handy höher und springe senkrecht zur Kameraachse."
        )

    quality = float(np.clip(
        0.35 * min(fit_hip.r2 / MIN_R2, 1.0)
        + 0.30 * (1.0 - min(disagreement / MAX_DISAGREE_M, 1.0))
        + 0.20 * min(n_air / 40.0, 1.0)
        + 0.15 * (conf if np.isfinite(conf) else 0.0), 0.0, 1.0))

    last = t.size - 1
    n1c = min(n1, last)
    return Result(
        h_e1=h_e1, h_e2=h_e2, h_e3=h_e3, h_e4=h_e4,
        spread=spread, disagreement=disagreement,
        t_f=t_f, v0=v0, v_land=v_land, delta=delta,
        t_takeoff=t0, t_land=t1,
        takeoff_frame=n0 + (t0 - t[n0]) / max(t[min(n0 + 1, last)] - t[n0], 1e-9),
        landing_frame=n1c + (t1 - t[n1c]) / max(t[min(n1c + 1, last)] - t[n1c], 1e-9),
        contact_time=contact_time, push_time=push_time, rsi=rsi,
        impulse=impulse, mean_force=mean_force, mass_kg=mass_kg,
        scale=s, a_px=fit_hip.a_px, r2_hip=fit_hip.r2, r2_com=fit_com.r2,
        rms_hip_px=fit_hip.rms_resid, cubic_t=fit_hip.cubic_t,
        n_flight_frames=n_air, mean_confidence=conf,
        shift_residual_px=td.shift_residual, fps_measured=td.fps,
        fit_c0=fit_hip.c0, fit_c1=fit_hip.c1, fit_c2=fit_hip.c2,
        fit_t_ref=fit_hip.t_ref,
        jump_type=jump_type, quality=quality, corrected=corrected, warnings=warn,
    )


def _measure_flight_time(td: TrackData, *, n0: int, n1: int, t0: float, t1: float,
                         t_f: float, jump_type: str, mass_kg: float | None,
                         corrected: bool, grund: str) -> Result:
    """Zweite Wahl: die Höhe allein aus der Flugzeit, `h = g*t_f^2/8`.

    Das ist die Methode, die jede andere Sprung-App benutzt, und sie hat genau
    eine Voraussetzung: gleiche Schwerpunkthöhe bei Absprung und Landung. Die
    ist verletzt (man springt gestreckt ab und landet gebeugt), **deshalb
    überschätzt E1 systematisch** - auf der sauberen Referenz um 3.0 cm, auf
    einem Anlaufsprung um 24.3 cm. Um wie viel es im Einzelfall ist, sagt nur
    die Parabel, und die ist hier gerade weggefallen. Der Wert kommt darum mit
    Ansage und nicht als Referenzwert.

    Was E1 dafür **nicht** braucht: einen Massstab, eine Pose, eine Perspektive.
    In der Formel steht keine einzige Pixelkoordinate. Genau deshalb überlebt
    sie, was E2 umbringt - verrauschte Keypoints, schräge Kameraachse,
    Massstabsänderung im Flug. Was sie **doch** braucht, sind die beiden
    Kontaktframes und eine richtige Zeitachse; steht davon etwas in Frage, hat
    `measure` schon vorher verweigert.

    Die Kontaktzeit bleibt (sie kommt aus demselben Signal), die Abdruckdauer
    fällt weg: die hängt am tiefsten Punkt der Hüfte - also an genau der
    Spur, die gerade als unbrauchbar gilt. Ohne sie gibt es keine mittlere
    Kraft. Der Impuls bleibt, weil v0 hier aus der Flugzeit kommt.
    """
    t = td.t
    h_e1 = G * t_f * t_f / 8.0
    v0 = G * t_f / 2.0          # dieselbe Symmetrieannahme wie E1, nicht mehr
    sel = slice(n0 + 1, n1 - 1)
    with np.errstate(invalid="ignore"):
        conf = (float(np.nanmean(td.confidence[sel]))
                if sel.stop > sel.start else float("nan"))

    k = n0 - 1
    while k > 0 and td.contact[k] >= 0.5:
        k -= 1
    ct = float(t0 - t[k + 1]) if n0 - (k + 1) >= 2 else None
    contact_time = ct if (ct is not None and ct <= 1.5) else None

    warn = _warnings(td, corrected=corrected)
    warn.insert(0, f"Die Flugbahn liess sich nicht auswerten: {grund} "
                   f"Die Höhe kommt deshalb allein aus der Flugzeit "
                   f"(h = g·t²/8). Sie überschätzt bauartbedingt, weil man "
                   f"gestreckt abspringt und gebeugt landet - auf sauberen "
                   f"Aufnahmen um 3 bis 6 cm, bei tiefer Landung deutlich mehr. "
                   f"Absprung, Landung und Flugzeit stimmen davon unberührt.")

    last = t.size - 1
    n1c = min(n1, last)
    return Result(
        h_e1=h_e1, h_e2=None, h_e3=None, h_e4=None,
        spread=None, disagreement=None,
        t_f=t_f, v0=v0, v_land=None, delta=None,
        t_takeoff=t0, t_land=t1,
        takeoff_frame=n0 + (t0 - t[n0]) / max(t[min(n0 + 1, last)] - t[n0], 1e-9),
        landing_frame=n1c + (t1 - t[n1c]) / max(t[min(n1c + 1, last)] - t[n1c], 1e-9),
        contact_time=contact_time, push_time=None, rsi=None,
        impulse=(mass_kg * v0) if mass_kg else None, mean_force=None,
        mass_kg=mass_kg,
        scale=None, a_px=None, r2_hip=None, r2_com=None, rms_hip_px=None,
        cubic_t=None, n_flight_frames=n1 - n0, mean_confidence=conf,
        shift_residual_px=td.shift_residual, fps_measured=td.fps,
        fit_c0=None, fit_c1=None, fit_c2=None, fit_t_ref=None,
        jump_type=jump_type, quality=None, mode="flugzeit", fallback_reason=grund,
        corrected=corrected, warnings=warn,
    )


def analyze(path: str | Path, *, jump_type: str = "cmj", mass_kg: float | None = None,
            seed: int = 0, scale: float | None = None, model: str = "full",
            track_out: str | Path | None = None,
            capture_fps: float | None = None) -> Result:
    """Bequemlichkeit für die CLI: extrahieren und messen in einem Zug."""
    from .extraction import extract

    td = extract(path, scale=scale, model=model, capture_fps=capture_fps)
    if track_out is not None:
        td.save(track_out)
    return measure(td, jump_type=jump_type, mass_kg=mass_kg, seed=seed)
