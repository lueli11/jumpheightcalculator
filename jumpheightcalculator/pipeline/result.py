"""Result - das Ergebnisobjekt von `measure`, mit allen vier Schätzern."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import PIPELINE_VERSION


@dataclass(frozen=True)
class Result:
    h_e1: float
    h_e2: float | None
    h_e3: float | None
    h_e4: float | None
    spread: float | None
    disagreement: float | None
    t_f: float
    v0: float | None
    v_land: float | None
    delta: float | None
    t_takeoff: float
    t_land: float
    takeoff_frame: float
    landing_frame: float
    contact_time: float | None     # nur beim Drop Jump definiert
    push_time: float | None               # tiefster Punkt der Gegenbewegung bis Absprung
    rsi: float | None
    impulse: float | None
    mean_force: float | None
    # Die Masse gehört ins Ergebnis, nicht nur in die Rechnung: ohne sie lässt
    # sich später nicht sagen, mit welchem Gewicht Impuls und Kraft entstanden.
    mass_kg: float | None
    scale: float | None
    a_px: float | None
    r2_hip: float | None
    r2_com: float | None
    rms_hip_px: float | None
    cubic_t: float | None
    n_flight_frames: int
    mean_confidence: float
    shift_residual_px: float
    fps_measured: float
    # Gefittete Parabel in Arbeitspixeln: y(t) = c0 + c1*(t-t_ref) + c2*(t-t_ref)^2.
    # Das Overlay zeichnet damit genau die Kurve, aus der die Zahl kommt - wer sie
    # sieht, kann der Zahl widersprechen.
    fit_c0: float | None
    fit_c1: float | None
    fit_c2: float | None
    fit_t_ref: float | None
    jump_type: str
    quality: float | None
    # "parabel" = der Referenzwert E2 aus dem Hüfte-Fit. "flugzeit" = die Bahn
    # war unbrauchbar, geblieben ist E1 aus den zwei Kontaktframes. E1 braucht
    # **keinen Massstab** (h = g*t^2/8) und ist damit gegen Pose-Rauschen,
    # Kamerawinkel und Perspektive immun - aber es überschätzt bauartbedingt
    # (R2), und um wie viel, sagt nur die Parabel. Darum zweite Wahl.
    mode: str = "parabel"
    # Warum die Bahn verworfen wurde - nur im Modus "flugzeit" gesetzt.
    fallback_reason: str | None = None
    corrected: bool = False
    warnings: list[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)
